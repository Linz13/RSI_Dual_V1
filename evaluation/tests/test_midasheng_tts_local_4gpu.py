import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import midasheng_tts_local_eval_4gpu as m


class FourGPUImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'old'
        self.dest = self.root / 'new'
        self.data = self.root / 'data.jsonl'
        self.data.write_text('{}\n')
        self.mos = self.root / 'mos'
        self.mos.write_bytes(b'mos')
        self.patches = [patch.object(m.base, 'DATA', self.data), patch.object(m.base, 'MOS', self.mos)]
        for p in self.patches: p.start()
        self.identity = {'completed_rounds': 6, 'code_round': 5, 'adapter': {'path': '/fixture/adapter'}}
        self.generation = m.base.expected_generation(self.identity, 'full')
        self.code = self.root / 'worker.py'
        self.code.write_text('# unchanged worker')
        m.base.write_json(self.source / 'local_run_identity.json', {
            **self.identity, 'generation': self.generation, 'source_sha256': {str(self.code): m.base.sha256_file(self.code)}})
        p = self.source / 'audios/0.wav'
        p.parent.mkdir()
        with wave.open(str(p), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b'\0' * 480)
        m.base.write_json(self.source / 'generation_manifest.json', {
            'generation_identity': self.generation, 'generation_identity_sha256': m.base.sha256_json(self.generation),
            'expected_samples': 1645, 'samples': [{'unique_id_eval': 0, 'status': 'generated', 'output_path': str(p)}]})

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_import_copies_audio_preserves_source_and_resumes(self):
        before = {str(p): m.base.sha256_file(p) for p in self.source.rglob('*') if p.is_file()}
        m.import_audio(self.source, self.dest, self.identity, ['4', '5'], {})
        m.import_audio(self.source, self.dest, self.identity, ['4', '5'], {})
        after = {str(p): m.base.sha256_file(p) for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual((self.dest / 'audios/0.wav').read_bytes(), (self.source / 'audios/0.wav').read_bytes())
        self.assertEqual(json.loads((self.dest / 'audio_import.json').read_text())['copied_samples'], 1)
        self.assertNotEqual((self.dest / 'audios/0.wav').stat().st_ino, (self.source / 'audios/0.wav').stat().st_ino)

    def test_reject_changed_source_worker(self):
        self.code.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Source code changed'):
            m.source_records(self.source, self.identity)

    def test_reject_wrong_checkpoint_or_generation_protocol(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
            m.source_records(self.source, {**self.identity, 'code_round': 8})
        p = self.source / 'generation_manifest.json'
        d = json.loads(p.read_text()); d['generation_identity']['seed'] = 99
        m.base.write_json(p, d)
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            m.source_records(self.source, self.identity)

    def test_refuse_resume_on_changed_gpu_identity(self):
        m.import_audio(self.source, self.dest, self.identity, ['4', '5'], {})
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            m.import_audio(self.source, self.dest, self.identity, ['6', '7'], {})

    def test_scoring_uses_two_visible_gpus_generation_unchanged(self):
        gen, local = m.base.commands(self.dest, self.identity, 'full')
        runner = m.FourGPURunner()
        with patch.object(m.base.Runner, 'execute') as execute:
            runner.execute(gen, {}, self.dest / 'generation.log')
            self.assertEqual(execute.call_args.args[0], gen)
            runner.execute(local, {}, self.dest / 'local.log')
            cmd = execute.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--workers') + 1], '2')
            self.assertEqual(cmd[cmd.index('--devices') + 1], 'cuda:0,cuda:1')
            self.assertFalse(any('judge' in arg for arg in cmd))


if __name__ == '__main__':
    unittest.main()
