import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from development_analysis import outcome
from development_recovery import durable
from development_guard import bounded_run


class RecoveryTests(unittest.TestCase):
    def test_joint_outcomes_preserve_uncertainty_and_missingness(self):
        self.assertEqual(outcome([.1,.2],[30,40],.1,50),'PASS')
        self.assertEqual(outcome([.1,.2],[45,55],.1,50),'UNCERTAIN')
        self.assertEqual(outcome([.01,.03],None,.1,50),'FAIL')
        self.assertEqual(outcome([.1,.2],None,.1,50),'NOT_EVALUATED')
        self.assertEqual(outcome(None,[51,55],.1,50),'FAIL')
        self.assertEqual(outcome([.09,.11],[30,40],.1,50),'UNCERTAIN')

    def test_durable_flush_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'result.json';data=json.dumps({'status':'PASS'}).encode();f.write_bytes(data)
            durable(p);self.assertEqual(f.read_bytes(),data)

    @unittest.skipUnless(os.name=='nt','Windows process-tree protection')
    def test_timeout_stops_the_owned_grandchild(self):
        import ctypes
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'grandchild.txt'
            code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)"
            with (Path(tmp)/'log.txt').open('w') as stream:
                with self.assertRaises(subprocess.TimeoutExpired):
                    bounded_run([sys.executable,'-c',code,str(path)],stream,3,dict(os.environ))
            self.assertTrue(path.exists())
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.restype=ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
            kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            handle=kernel.OpenProcess(0x1000,False,int(path.read_text()))
            if handle:
                exit_code=ctypes.c_ulong();kernel.GetExitCodeProcess(handle,ctypes.byref(exit_code));kernel.CloseHandle(handle)
                self.assertNotEqual(exit_code.value,259)


if __name__=='__main__':unittest.main()
