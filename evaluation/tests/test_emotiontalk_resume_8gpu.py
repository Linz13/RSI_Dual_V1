import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emotiontalk_resume_8gpu as et


class ResumeTests(unittest.TestCase):
    def rows(self, n=32):
        return [{'id': str(i//4), 'task': et.TASKS[i%4], 'prompt': 'prompt', 'audio_path': 'audio.wav'} for i in range(n)]

    def event(self, row, run_id='parent', text='prediction'):
        return {'id': row['id'], 'task': row['task'], 'run_id': run_id, 'status': 'ok', 'prediction': text}

    def test_partition_preserves_all_remaining_once_and_balances_tasks(self):
        rows = self.rows(7716)
        done = {et.key(r) for r in rows[:3037]}
        buckets = et.partition(rows, done)
        flat = [et.key(r) for b in buckets for r in b]
        self.assertEqual(len(flat), 7716-3037)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(set(flat).isdisjoint(done))
        self.assertEqual(set(flat) | done, {et.key(r) for r in rows})
        self.assertLessEqual(max(map(len,buckets))-min(map(len,buckets)), 4)

    def test_only_interrupted_final_line_can_be_recovered(self):
        row = self.event(self.rows(1)[0])
        raw = et.encode_rows([row])
        self.assertEqual(et.parse_events(raw+b'{"run_id":')[0], [row])
        self.assertEqual(et.parse_events(raw.rstrip(b'\n'))[0], [row])
        for broken in (raw+b'broken\n', b'broken\n'+raw):
            with self.assertRaises(ValueError):
                et.parse_events(broken)

    def test_merge_preserves_old_predictions_and_parent_identity(self):
        rows = self.rows(12)
        old = [self.event(r, text='old-'+str(i)) for i,r in enumerate(rows[:3])]
        shards = []
        for i, bucket in enumerate(et.partition(rows, {et.key(r) for r in rows[:3]})):
            if bucket:
                shards.append(([self.event(r,str(i)) for r in bucket],str(i),{et.key(r) for r in bucket}))
        merged = et.merged_events(old, shards, rows, 'parent')
        self.assertEqual(merged[:3], old)
        self.assertEqual([et.key(r) for r in merged], [et.key(r) for r in rows])
        self.assertTrue(all(r['run_id']=='parent' for r in merged))

    def test_merge_rejects_missing_overlap_and_wrong_identity(self):
        rows = self.rows(4)
        for old, events, child_keys in (
            ([], [self.event(r,'child') for r in rows[:3]], {et.key(r) for r in rows}),
            ([self.event(rows[0])], [self.event(r,'child') for r in rows], {et.key(r) for r in rows}),
            ([], [self.event(r,'wrong') for r in rows], {et.key(r) for r in rows}),
        ):
            with self.subTest(old=old,events=events), self.assertRaises(ValueError):
                et.merged_events(old, [(events,'child',child_keys)], rows, 'parent')

    def test_conflicting_duplicate_predictions_are_rejected(self):
        row=self.rows(1)[0]
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            et.successful([self.event(row,text='one'),self.event(row,text='two')],'parent',{et.key(row)})

    def test_snapshot_merge_and_resume_end_to_end_without_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder);work=source/'resume_8gpu';rows=self.rows(16)
            original=et.encode_rows([self.event(r,text='original') for r in rows[:3]])+b'{"interrupted":'
            (source/'events.jsonl').write_bytes(original)
            meta={'run_id':'parent'}
            plan=et.prepare(work,source,rows,meta,'inventory')
            self.assertEqual(plan['preserved_successes'],3)
            self.assertEqual((work/'original_events.bin').read_bytes(),original)
            def fake_identity(c,manifest,selected):
                return {'manifest_hash':et.digest(manifest.read_bytes()),'model':'same-model'}
            for spec in plan['shards']:
                if not spec['count']:continue
                manifest=Path(spec['manifest']);selected=et.legacy.rows(manifest)
                expected=fake_identity(None,manifest,selected);run_id=et.sha256_json(expected)
                out=manifest.parent/'output';out.mkdir()
                et.legacy.write_json(out/'run_metadata.json',{'identity':expected,'run_id':run_id})
                et.atomic_bytes(out/'events.jsonl',et.encode_rows([self.event(r,run_id) for r in selected]))
            with mock.patch.object(et,'identity',side_effect=fake_identity):
                et.merge(work,source,plan,rows,meta,None)
                first=(source/'events.jsonl').read_bytes()
                self.assertEqual(et.legacy.rows(source/'predictions.jsonl')[:3],
                                 [{k:r[k] for k in ('id','task','prediction')} for r in et.parse_events(original)[0]])
                # Simulate interruption after replacing events but before predictions.
                (source/'predictions.jsonl').unlink()
                et.merge(work,source,plan,rows,meta,None)
                self.assertEqual((source/'events.jsonl').read_bytes(),first)
                self.assertEqual(len(et.legacy.rows(source/'predictions.jsonl')),16)
                et.atomic_bytes(source/'events.jsonl',first+et.encode_rows([self.event(rows[0],text='changed')]))
                with self.assertRaisesRegex(ValueError,'Parent events changed'):
                    et.merge(work,source,plan,rows,meta,None)
            self.assertEqual(et.prepare(work,source,rows,meta,'inventory'),plan)


if __name__=='__main__':
    unittest.main()
