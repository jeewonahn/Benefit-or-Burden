"""The official 400k vocabulary has a real extra <unk>, never arbitrary rows."""
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_model_assets import validate_glove_text
from prepare_digat_assets import validate_glove_vocabulary
class GloveUnknownTests(unittest.TestCase):
    def inspect(self,text):
        with tempfile.TemporaryDirectory()as root:
            path=Path(root)/'vectors.txt';path.write_text(text,encoding='utf-8')
            return validate_glove_text(path,dimensions=2,expected_rows=2)
    def test_real_final_unknown_is_additional(self):
        result=self.inspect('one 1 2\ntwo 3 4\n<unk> 0 0\n')
        self.assertEqual((result['rows'],result['ordinary_rows'],result['unknown_rows']),(3,2,1))
    def test_without_unknown_remains_supported(self):
        self.assertEqual(self.inspect('one 1 2\ntwo 3 4\n')['rows'],2)
    def test_arbitrary_extra_ordinary_row_rejected(self):
        with self.assertRaises(ValueError):self.inspect('one 1 2\ntwo 3 4\nextra 0 0\n')
    def test_duplicate_unknown_rejected(self):
        with self.assertRaises(ValueError):self.inspect('one 1 2\ntwo 3 4\n<unk> 0 0\n<unk> 0 0\n')
    def test_nonfinal_unknown_rejected(self):
        with self.assertRaises(ValueError):self.inspect('<unk> 0 0\none 1 2\ntwo 3 4\n')
    def test_digat_accepts_certified_real_vocabulary(self):
        validate_glove_vocabulary(dict(dimensions=300,rows=400001,ordinary_rows=400000,unknown_rows=1))
        validate_glove_vocabulary(dict(dimensions=300,rows=400000))
    def test_digat_rejects_uncertified_extra_row(self):
        with self.assertRaises(ValueError):validate_glove_vocabulary(dict(dimensions=300,rows=400001))
if __name__=='__main__':unittest.main()