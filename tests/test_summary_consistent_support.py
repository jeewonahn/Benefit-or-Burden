"""Synthetic routing tests for eligible IMP versus preserved legacy summary support."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE=Path(__file__).resolve().parents[1]/'scripts/summarize_converged_experiments.py'
PROFILE='complete_proxy_strict_request_consistent_v1'
PROTOCOLS=('imp','rs30','rs90','rs150','total')


def function(profile,paths):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    target=next(item for item in tree.body if isinstance(item,ast.FunctionDef)and item.name=='evaluation_expectations')
    namespace={'Path':Path,'PROTOCOLS':PROTOCOLS,'CONSISTENT_PROFILE':PROFILE,
        'validate_candidate_profile':mock.Mock(return_value=profile),'evaluation_paths':mock.Mock(return_value=paths)}
    exec(compile(ast.Module(body=[target],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace['evaluation_expectations'],namespace


class SummarySupportTests(unittest.TestCase):
    def fixture(self,root):
        names={'imp':'behaviors.tsv','rs30':'behaviors_rs30.tsv','rs90':'behaviors_rs90.tsv','rs150':'behaviors_rs150.tsv'}
        dataset={'output_files':{},'statistics':{}}
        for protocol,name in names.items():
            dataset['output_files']['test/'+name]={'sha256':'legacy_'+protocol}
            dataset['statistics']['test/'+name]={'rows':4 if protocol=='imp' else 3,'candidates':20}
        dataset['output_files']['test/behaviors_imp_eligible.tsv']={'sha256':'eligible_imp'}
        dataset['statistics']['test/behaviors_imp_eligible.tsv']={'rows':3,'candidates':18}
        paths={p:root/'test'/('behaviors_imp_eligible.tsv'if p=='imp'else names[p])for p in names}
        paths['total']=root/'test/behaviors_total.tsv'
        candidate={'output':{'sha256':'complete_total'},'statistics':{'rows':3,'candidates':200}}
        return dataset,candidate,paths
    def test_complete_profile_uses_eligible_imp_and_identical_gt_support(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary).resolve();dataset,candidate,paths=self.fixture(root)
            call,namespace=function({'profile':PROFILE},paths);inputs,sizes=call(root,dataset,candidate)
            self.assertEqual(inputs['imp'],'eligible_imp');self.assertEqual(sizes['imp']['rows'],3)
            self.assertEqual(set(inputs),set(PROTOCOLS));self.assertEqual(inputs['total'],'complete_total')
            self.assertEqual({row['rows']for row in sizes.values()},{3})
            namespace['evaluation_paths'].assert_called_once()
    def test_legacy_keeps_original_imp_filename_and_does_not_route_certificate_paths(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary).resolve();dataset,candidate,paths=self.fixture(root)
            call,namespace=function(None,paths);inputs,sizes=call(root,dataset,candidate)
            self.assertEqual(inputs['imp'],'legacy_imp');self.assertEqual(sizes['imp']['rows'],4)
            namespace['evaluation_paths'].assert_not_called()
    def test_profile_validation_failure_is_not_silently_treated_as_legacy(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary).resolve();dataset,candidate,paths=self.fixture(root)
            call,namespace=function({'profile':PROFILE},paths)
            namespace['validate_candidate_profile'].side_effect=ValueError('certificate differs')
            with self.assertRaisesRegex(ValueError,'certificate differs'):call(root,dataset,candidate)
    def test_missing_eligible_statistics_rejected_instead_of_using_master(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary).resolve();dataset,candidate,paths=self.fixture(root)
            del dataset['statistics']['test/behaviors_imp_eligible.tsv'];call,_=function({'profile':PROFILE},paths)
            with self.assertRaises(KeyError):call(root,dataset,candidate)


if __name__=='__main__':unittest.main()
