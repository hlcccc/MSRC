"""Contract tests for bounded, development-only response-prefix annotations."""

from collections import Counter
from copy import deepcopy
import hashlib
import random
import unittest
from unittest.mock import patch

from safety_lab.prefix_data import (
    parse_response_safety,
    prefix_positions,
    select_development,
    validate_prefix_labels,
)


SUITES = ("SD", "SD_TYPO", "TYPO", "benign")
ROLES = ("fit", "calibration", "test")
STATUSES = {"safe": 0, "unsafe": 1, "controversial": None, "unparsed": None}


def partition(fit=5, calibration=3, test=2):
    rows, roles = [], {role: [] for role in ROLES}
    for role, count in zip(ROLES, (fit, calibration, test)):
        for suite in SUITES:
            for index in range(count):
                rid = f"{role}/{suite}/{index}"
                rows.append(dict(id=rid, split="test" if role == "test" else "dev",
                                 suite=suite, label=int(suite != "benign")))
                roles[role].append(rid)
    return rows, roles


def selected_rows():
    rows, roles = partition()
    return select_development(rows, roles, fit_per_suite=1, calibration_per_suite=1)


def annotation(row, status="safe", length=40, position=16):
    raw = "No safety field was returned." if status == "unparsed" else (
        "**Safety:** " + status.title() + "\nReason: independently judged prefix.")
    return dict(id=row["id"], role=row["role"], position=position,
                generated_tokens=length, label=STATUSES[status], status=status,
                label_source="response_prefix_judge", judge_raw=raw,
                prefix_sha256=hashlib.sha256(b"observed prefix").hexdigest(),
                judge_prompt_sha256=hashlib.sha256(b"judge prompt").hexdigest())


def valid_annotations(selected):
    return [annotation(next(row for row in selected if row["role"] == role))
            for role in ("fit", "calibration")]


class SelectionTests(unittest.TestCase):
    def test_default_caps_and_exact_hash_order_per_role_and_suite(self):
        rows, roles = partition(fit=70, calibration=20)
        actual = select_development(rows, roles)
        expected = []
        for role, limit in (("fit", 64), ("calibration", 16)):
            for suite in SUITES:
                stratum = [row for row in rows
                           if row["id"] in roles[role] and row["suite"] == suite]
                stratum.sort(key=lambda row: (
                    hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), row["id"]))
                expected.extend(dict(row, role=role) for row in stratum[:limit])
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 320)
        self.assertTrue(all(row["split"] == "dev" for row in actual))
        self.assertTrue(set(row["id"] for row in actual).isdisjoint(roles["test"]))

    def test_small_strata_are_capped_not_required_to_fill_limits(self):
        rows, roles = partition(fit=2, calibration=1)
        selected = select_development(rows, roles)
        self.assertEqual(Counter(row["role"] for row in selected),
                         {"fit": 8, "calibration": 4})

    def test_input_role_order_and_iterators_do_not_change_selection(self):
        rows, roles = partition()
        expected = select_development(rows, roles, 3, 2)
        random.Random(37).shuffle(rows)
        reordered = {role: iter(reversed(roles[role])) for role in reversed(ROLES)}
        self.assertEqual(select_development(iter(rows), reordered, 3, 2), expected)
        self.assertEqual(select_development(rows, {k: set(v) for k, v in roles.items()},
                                            3, 2), expected)

    def test_rows_are_copied_without_mutating_inputs_or_roles(self):
        rows, roles = partition()
        original_rows, original_roles = deepcopy(rows), deepcopy(roles)
        selected = select_development(rows, roles, 1, 1)
        self.assertEqual(rows, original_rows)
        self.assertEqual(roles, original_roles)
        for row in selected:
            source = next(item for item in rows if item["id"] == row["id"])
            self.assertIsNot(row, source)
            self.assertEqual({k: v for k, v in row.items() if k != "role"}, source)
        selected[0]["suite"] = "changed"
        self.assertEqual(rows, original_rows)

    def test_hash_ties_use_id_order(self):
        rows, roles = partition(fit=12, calibration=12)
        with patch("safety_lab.prefix_data.hashlib.sha256") as digest:
            digest.return_value.hexdigest.return_value = "0" * 64
            actual = select_development(reversed(rows), roles, 2, 2)
        expected = []
        for role in ("fit", "calibration"):
            for suite in SUITES:
                expected.extend(sorted(rid for rid in roles[role]
                                       if f"/{suite}/" in rid)[:2])
        self.assertEqual([row["id"] for row in actual], expected)

    def test_id_hashing_is_utf8(self):
        rows, roles = partition(fit=1, calibration=1)
        old = rows[0]["id"]
        rows[0]["id"] = "\u524d\u7f00/\u00e9"
        roles["fit"][roles["fit"].index(old)] = rows[0]["id"]
        self.assertIn(rows[0]["id"], [r["id"] for r in select_development(rows, roles)])

    def test_limits_are_strict_positive_integers(self):
        rows, roles = partition()
        for name in ("fit_per_suite", "calibration_per_suite"):
            for value in (0, -1, True, False, 1.0, "1", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    select_development(rows, roles, **{name: value})

    def test_duplicate_ids_are_rejected_even_outside_selected_rows(self):
        rows, roles = partition()
        for row in (rows[0], rows[-1]):
            with self.subTest(id=row["id"]), self.assertRaises(ValueError):
                select_development(rows + [dict(row)], roles, 1, 1)

    def test_invalid_or_missing_row_identity_is_rejected(self):
        for field in ("id", "split", "suite"):
            rows, roles = partition()
            del rows[-1][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_development(rows, roles)
        for rid in (None, "", 1, True, []):
            rows, roles = partition()
            rows[0]["id"] = rid
            with self.subTest(id=rid), self.assertRaises(ValueError):
                select_development(rows, roles)

    def test_duplicates_within_each_role_are_rejected(self):
        for role in ROLES:
            rows, roles = partition()
            roles[role].append(roles[role][0])
            with self.subTest(role=role), self.assertRaises(ValueError):
                select_development(rows, roles)

    def test_overlapping_roles_are_rejected(self):
        for left, right in (("fit", "calibration"), ("fit", "test"),
                            ("calibration", "test")):
            rows, roles = partition()
            roles[right].append(roles[left][0])
            with self.subTest(pair=(left, right)), self.assertRaises(ValueError):
                select_development(rows, roles)

    def test_role_ids_must_cover_exactly_all_input_ids(self):
        for role in ROLES:
            rows, roles = partition()
            roles[role].pop()
            with self.subTest(missing_assignment=role), self.assertRaises(ValueError):
                select_development(rows, roles)
            rows, roles = partition()
            roles[role].append("not-in-inputs")
            with self.subTest(missing_row=role), self.assertRaises(ValueError):
                select_development(rows, roles)

    def test_role_mapping_and_role_ids_are_validated(self):
        rows, roles = partition()
        invalid = [None, {}, dict(roles, other=[])]
        invalid.extend({key: value for key, value in roles.items() if key != role}
                       for role in ROLES)
        for role in ROLES:
            for value in (None, "fit/SD/0", [""], [1], [[]]):
                invalid.append(dict(roles, **{role: value}))
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                select_development(rows, mapping)

    def test_split_and_preexisting_role_must_match_assignment(self):
        for role in ROLES:
            for split in ("train", "", None, "dev" if role == "test" else "test"):
                rows, roles = partition()
                row = next(row for row in rows if row["id"] in roles[role])
                row["split"] = split
                with self.subTest(role=role, split=split), self.assertRaises(ValueError):
                    select_development(rows, roles)
            rows, roles = partition()
            rows[0]["role"] = role
            if role == "fit":
                self.assertTrue(select_development(rows, roles))
            else:
                with self.subTest(role=role), self.assertRaises(ValueError):
                    select_development(rows, roles)

    def test_unknown_suites_are_rejected_even_on_test_rows(self):
        for value in ("OTHER", "sd", "BENIGN", "", None, []):
            rows, roles = partition()
            rows[-1]["suite"] = value
            with self.subTest(suite=value), self.assertRaises(ValueError):
                select_development(rows, roles)

    def test_every_development_role_suite_stratum_must_be_nonempty(self):
        for role in ("fit", "calibration"):
            for suite in SUITES:
                rows, roles = partition()
                removed = {r["id"] for r in rows
                           if r["id"] in roles[role] and r["suite"] == suite}
                rows = [r for r in rows if r["id"] not in removed]
                roles[role] = [rid for rid in roles[role] if rid not in removed]
                with self.subTest(role=role, suite=suite), self.assertRaises(ValueError):
                    select_development(rows, roles)


class PrefixPositionTests(unittest.TestCase):
    def test_all_bounded_lengths_have_exact_unique_ascending_positions(self):
        for length in range(129):
            with self.subTest(length=length):
                expected = sorted({min(n, length) for n in (16, 32, 64, 128)} - {0})
                self.assertEqual(prefix_positions(length), expected)

    def test_lengths_are_strict_bounded_integers(self):
        for value in (-1, 129, True, False, 16.0, "16", None, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prefix_positions(value)


class SafetyParsingTests(unittest.TestCase):
    def test_exact_fields_case_whitespace_and_markdown_bold(self):
        for status, label in STATUSES.items():
            if status == "unparsed":
                continue
            for field in (f"Safety: {status.title()}", f"  sAfEtY : {status.upper()}\t",
                          f"**Safety:** {status.title()}",
                          f"**Safety**: **{status.title()}**",
                          f"**Safety: {status.title()}**",
                          f"__Safety:__ __{status.title()}__"):
                with self.subTest(field=field):
                    self.assertEqual(parse_response_safety(field),
                                     dict(label=label, status=status))

    def test_explanation_words_never_override_a_field(self):
        text = ("The unsafe request contains controversial terms.\r\n"
                "**Safety:** Safe\r\nReason: unsafe and controversial are request words.")
        self.assertEqual(parse_response_safety(text), dict(label=0, status="safe"))

    def test_missing_partial_inline_and_explanation_matches_are_unparsed(self):
        for text in ("", "Safe", "The response is unsafe.", "Reason: Safety: Unsafe",
                     "Response Safety: Unsafe", "Not Safety: Safe", "Unsafety: Safe",
                     '"Safety: Safe"', "Safety: Safely", "Safety: Unsafeish",
                     "Safety: Safe or Unsafe", "Safety: Unknown", "Safety: 0",
                     "Safety: Safe because it refuses.", "**Safety: Safe", "Safety: Safe**"):
            with self.subTest(text=text):
                self.assertEqual(parse_response_safety(text),
                                 dict(label=None, status="unparsed"))

    def test_contradictory_fields_are_unparsed_regardless_of_order(self):
        fields = ("Safety: Safe", "**Safety:** Unsafe", "Safety: Controversial")
        for left in fields:
            for right in fields:
                if left != right:
                    with self.subTest(left=left, right=right):
                        self.assertEqual(parse_response_safety(left + "\n" + right),
                                         dict(label=None, status="unparsed"))

    def test_repeated_agreeing_fields_are_accepted(self):
        self.assertEqual(parse_response_safety("Safety: Safe\n**Safety:** SAFE"),
                         dict(label=0, status="safe"))

    def test_non_text_is_rejected(self):
        for text in (None, 0, b"Safety: Safe", []):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response_safety(text)


class LabelValidationTests(unittest.TestCase):
    def setUp(self):
        self.selected = selected_rows()
        self.records = valid_annotations(self.selected)

    def assert_bad_record(self, **changes):
        records = deepcopy(self.records)
        records[0].update(changes)
        with self.assertRaises(ValueError):
            validate_prefix_labels(records, self.selected)

    def test_counts_by_role_label_and_status_include_excluded_annotations(self):
        records = []
        for role in ("fit", "calibration"):
            row = next(r for r in self.selected if r["role"] == role)
            records.extend(annotation(row, status, length=128, position=position)
                           for status, position in zip(STATUSES, (16, 32, 64, 128)))
        expected = {role: {"total": 4, "by_label": {0: 1, 1: 1, None: 2},
                           "by_status": {status: 1 for status in STATUSES}}
                    for role in ("fit", "calibration")}
        self.assertEqual(validate_prefix_labels(records, self.selected), expected)

    def test_single_class_and_small_partial_annotation_sets_are_allowed(self):
        counts = validate_prefix_labels(self.records, self.selected)
        for role in ("fit", "calibration"):
            self.assertEqual(counts[role]["total"], 1)
            self.assertEqual(counts[role]["by_label"], {0: 1, 1: 0, None: 0})
            self.assertEqual(counts[role]["by_status"]["unparsed"], 0)

    def test_no_binary_annotations_are_required_by_this_helper(self):
        for record in self.records:
            record.update(label=None, status="unparsed", judge_raw="")
        self.assertEqual(validate_prefix_labels(self.records, self.selected)["fit"]["total"], 1)

    def test_response_labels_need_not_equal_request_labels(self):
        for row in self.selected:
            row["label"] = 1
        self.assertTrue(validate_prefix_labels(self.records, self.selected))
        self.records[0].update(label=1, status="unsafe", judge_raw="Safety: Unsafe")
        self.assertTrue(validate_prefix_labels(self.records, self.selected))

    def test_all_legal_prefixes_including_short_and_capped_terminal_positions(self):
        for length in (1, 15, 16, 17, 32, 40, 64, 65, 127, 128):
            records = [annotation(row, length=length, position=position)
                       for row in (self.selected[0], self.selected[-1])
                       for position in prefix_positions(length)]
            with self.subTest(length=length):
                self.assertEqual(validate_prefix_labels(records, self.selected)["fit"]["total"],
                                 len(prefix_positions(length)))

    def test_validation_is_order_independent_accepts_iterators_and_does_not_mutate(self):
        before = deepcopy((self.records, self.selected))
        expected = validate_prefix_labels(self.records, self.selected)
        self.assertEqual(validate_prefix_labels(iter(reversed(self.records)),
                                              iter(reversed(self.selected))), expected)
        self.assertEqual((self.records, self.selected), before)

    def test_unexpected_test_and_duplicate_record_keys_are_rejected(self):
        for rid in ("unexpected", "test/SD/0", None, "", 1, []):
            with self.subTest(id=rid):
                self.assert_bad_record(id=rid)
        with self.assertRaises(ValueError):
            validate_prefix_labels(self.records + [dict(self.records[0])], self.selected)

    def test_roles_must_be_development_roles_and_match_selected_rows(self):
        for role in ("test", "dev", "train", "calibration", "", None, []):
            with self.subTest(role=role):
                self.assert_bad_record(role=role)

    def test_selected_rows_must_be_unique_known_development_rows(self):
        with self.assertRaises(ValueError):
            validate_prefix_labels(self.records, self.selected + [dict(self.selected[0])])
        for changes in ({"split": "test"}, {"split": "train"}, {"role": "test"},
                        {"role": "calibration"}, {"role": []}, {"suite": "OTHER"}):
            selected = deepcopy(self.selected)
            selected[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_prefix_labels(self.records, selected)
        for field in ("id", "split", "suite", "role"):
            selected = deepcopy(self.selected)
            del selected[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_prefix_labels(self.records, selected)

    def test_both_annotation_roles_must_be_present(self):
        for records in ([], self.records[:1], self.records[1:]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_prefix_labels(records, self.selected)

    def test_invalid_generated_lengths_and_positions_are_rejected(self):
        for length in (-1, 0, 129, True, False, 40.0, "40", None):
            with self.subTest(length=length):
                self.assert_bad_record(generated_tokens=length)
        for position in (-1, 0, 1, 17, 39, 41, 64, 128, True, False, 16.0, "16", None):
            with self.subTest(position=position):
                self.assert_bad_record(position=position)

    def test_labels_require_exact_binary_integers_or_none(self):
        for label in (-1, 2, True, False, 0.0, 1.0, "0", "1", "None", [], {}):
            with self.subTest(label=label):
                self.assert_bad_record(label=label)

    def test_label_status_and_raw_judge_fields_must_agree(self):
        for status in ("SAFE", "unknown", "", None, []):
            with self.subTest(status=status):
                self.assert_bad_record(status=status)
        for changes in ({"label": 1}, {"label": None}, {"status": "unsafe"},
                        {"label": None, "status": "controversial"},
                        {"label": None, "status": "unparsed"},
                        {"judge_raw": "Safety: Unsafe"},
                        {"judge_raw": "Safety: Safe\nSafety: Unsafe"}):
            with self.subTest(changes=changes):
                self.assert_bad_record(**changes)
        self.records[0].update(label=None, status="unparsed",
                               judge_raw="Safety: Safe\nSafety: Unsafe")
        self.assertTrue(validate_prefix_labels(self.records, self.selected))

    def test_request_label_substitution_and_missing_provenance_are_rejected(self):
        for source in ("request", "request_label", "input_label", "", None):
            with self.subTest(source=source):
                self.assert_bad_record(label_source=source)
        for field in tuple(self.records[0]):
            records = deepcopy(self.records)
            del records[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_prefix_labels(records, self.selected)
        for raw in (None, 0, [], b"Safety: Safe"):
            with self.subTest(raw=raw):
                self.assert_bad_record(judge_raw=raw)

    def test_sha256_fields_require_exact_64_character_hex_digests(self):
        for field in ("prefix_sha256", "judge_prompt_sha256"):
            for digest in (None, 123, [], "", "0" * 63, "0" * 65, "g" * 64,
                           " " + "0" * 64, "0" * 64 + "\n", "\uff10" * 64):
                with self.subTest(field=field, digest=digest):
                    self.assert_bad_record(**{field: digest})
            records = deepcopy(self.records)
            records[0][field] = records[0][field].upper()
            self.assertTrue(validate_prefix_labels(records, self.selected))

    def test_non_mapping_rows_and_records_fail_closed(self):
        rows, roles = partition()
        for row in (None, 1, "not a row", []):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    select_development(rows + [row], roles)
                with self.assertRaises(ValueError):
                    validate_prefix_labels(self.records + [row], self.selected)
                with self.assertRaises(ValueError):
                    validate_prefix_labels(self.records, self.selected + [row])


if __name__ == "__main__":
    unittest.main()
