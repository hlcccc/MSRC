"""The supplied six-cell inventory is independent of selected evaluation models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import factual, factual_parameters as fp


class SuppliedScopeContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.directory.name)
        self.path = self.root/'parameters.json'
        self.protocol = {k: None for k in fp.METHOD_KEYS}
        self.protocol['collector_entry'] = 'scripts/collect_evidence_fitted.py'
        state = dict(signals=factual.SIGNALS,live=[True]+[False]*(len(factual.SIGNALS)-1),
            train_ids=['dev'],converged=True,fitting_scope='development only',
            fill=[0.]*len(factual.SIGNALS),mean=[0.],std=[1.],coef=[0.,1.],
            warning_threshold=dict(value=.5,grid_points=201,tie_rule='highest',
                selected_on='development only',test_used_for_selection=False))
        self.value = dict(format=fp.FORMAT,method={k:self.protocol[k] for k in fp.METHOD_KEYS},
            datasets={'pope':'data','textvqa':'data'},states={m+'_'+d:deepcopy(state)
                for m in ('llava','qwen','internvl') for d in ('pope','textvqa')})

    def tearDown(self):
        self.directory.cleanup()

    def load(self,models=None):
        self.path.write_text(json.dumps(self.value),encoding='utf-8')
        with patch.object(factual,'MODELS',models or {'internvl':'not-loaded'}), \
                patch.object(fp,'validate_code_contract') as contract, \
                patch.object(fp,'read_rows',return_value=[dict(id='dev',acceptance_split='dev')]), \
                patch.object(fp,'identity',return_value='data'):
            result = fp.load_package(self.path,self.root,self.protocol)
        contract.assert_called_once_with(self.value,self.protocol['collector_entry'])
        return result

    def test_complete_six_state_package_loads_with_only_internvl_selected(self):
        self.assertEqual(self.load(),self.value)

    def test_complete_six_state_package_loads_without_llava_weights(self):
        self.assertEqual(self.load({'qwen':'donor','internvl':'selected'}),self.value)

    def test_original_three_model_scope_still_loads(self):
        self.assertEqual(self.load({m:'not-loaded' for m in ('llava','qwen','internvl')}),self.value)

    def test_missing_unused_state_is_rejected(self):
        del self.value['states']['llava_pope']
        with self.assertRaisesRegex(ValueError,'all six'):
            self.load()

    def test_extra_state_is_rejected(self):
        self.value['states']['other_pope']=deepcopy(self.value['states']['internvl_pope'])
        with self.assertRaisesRegex(ValueError,'all six'):
            self.load()

    def test_unused_parameter_state_is_still_numerically_validated(self):
        self.value['states']['qwen_textvqa']['std']=[0.]
        with self.assertRaisesRegex(ValueError,'nonpositive fitted scale'):
            self.load()

    def test_unused_state_training_identity_is_still_validated(self):
        self.value['states']['llava_pope']['train_ids']=['different']
        with self.assertRaisesRegex(ValueError,'training identity differs'):
            self.load()


if __name__ == '__main__':
    unittest.main(verbosity=2)
