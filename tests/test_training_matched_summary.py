"""CPU-only checks for training-specific checkpoint-selection provenance."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_converged_experiments as summary

class MatchedSummaryTests(unittest.TestCase):
    def expected(self,protocol="rs50"):
        return dict(protocol=protocol,sha256="dev-hash",rows=1,candidates=2,candidate_profile={"sha256":"certificate-hash"})
    def manifest(self,protocol="rs50"):
        return dict(model="NAML",train_type=protocol,selection=dict(protocol=protocol,file={"sha256":"dev-hash"},
                    matched_candidate_profile={"sha256":"certificate-hash"}),
                    epoch_records=[{"dev":{"input":{"sha256":"dev-hash"},"rows":1,"candidates":2}}])
    def test_matching_dev_is_accepted(self):
        summary.verify_development_records(self.manifest(),self.expected())
    def test_old_imp_dev_rands_model_is_rejected(self):
        m=self.manifest();m["selection"]["protocol"]="imp"
        with self.assertRaisesRegex(ValueError,"validation input/protocol"):
            summary.verify_development_records(m,self.expected())
    def test_mixed_candidate_certificate_is_rejected(self):
        m=self.manifest();m["selection"]["matched_candidate_profile"]={"sha256":"old"}
        with self.assertRaisesRegex(ValueError,"candidate profile"):
            summary.verify_development_records(m,self.expected())
    def test_original_imp_digat_can_be_reused_without_relabelling(self):
        m=dict(model="DIGAT",train_type="imp",profile={"dev_protocol":"imp"},
               input_binding={"inputs":{"files":[{"path":"/source/dev/behaviors.tsv","sha256":"dev-hash"}]}},
               epoch_records=[{"dev":{"input":{"sha256":"dev-hash"},"rows":1,"candidates":2}}])
        summary.verify_development_records(m,self.expected("imp"))
        self.assertNotIn("selection",m)
    def test_legacy_rands_cannot_use_imp_compatibility_branch(self):
        m=dict(model="DIGAT",train_type="rs50",profile={"dev_protocol":"imp"},input_binding={"inputs":{"files":[]}},epoch_records=[])
        with self.assertRaises(ValueError):summary.verify_development_records(m,self.expected())
    def test_rands_dev_counts_use_the_selected_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"dev").mkdir()
            (root/"dev/behaviors_rs50.tsv").write_text("1\tu\t01/01/2020 01:00:00 AM\t\tN1-1 N2-0\n",encoding="utf-8")
            with patch.object(summary,"validate_candidate_profile",return_value={"sha256":"proof"}):
                e=summary.development_expectation(root,{}, "rs50")
            self.assertEqual((e["rows"],e["candidates"]),(1,2))
            self.assertEqual(e["protocol"],"rs50")
if __name__=="__main__":unittest.main()

