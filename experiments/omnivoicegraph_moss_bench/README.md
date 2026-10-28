# OmniVoiceGraph MOSS-style objective benchmark

This bundle follows the evaluation contract in MOSS Transcribe Diarize (arXiv:2601.01554):

* `CER`: character error rate after removing speaker/timestamp markup.
* `cpCER`: concatenated minimum-permutation character error rate over speaker streams.
* `delta_cp = cpCER - CER`: transcript-attributed degradation caused by speaker attribution.

The canonical MOSS table names four datasets: AISHELL-4, Alimeeting, Podcast, and Movies. The official SGLang Omni benchmark instructions identify the long-form aliases `aishell4_long` and `googletime`, plus `movies800times`, and state that those benchmark bundles are currently private-license. Public Hugging Face mirrors for AISHELL4, Movies800Time, and GoogleTime are now staged locally, but their cards do not declare redistribution terms; they are marked `unverified_public_mirror` and must not be included in a public release until the MOSS team confirms provenance and licensing. The public AliMeeting mirror contains no transcript field, but a separately published DiariST reference bundle restores the Chinese CER/cpCER references for the same 20 test recording IDs.

The current OmniVoiceGraph checkpoint is evaluated in an English-only lane. AISHELL-4 and AliMeeting are Chinese speech benchmarks and are excluded; an English translation in AliMeeting is not an English ASR target. Podcast/GoogleTime is English. Movies is mixed-language, so `make_english_only_manifests.py` keeps only rows whose references contain Latin-script alphabetic text and no other script: 281/800 timed rows and 290/800 no-timestamp control rows. This is a conservative script gate, not a full language-identification claim; every published result must include the filtered-manifest receipt and denominator.

AliMeeting reference provenance: `download_alimeeting_references.py` follows the DiariST repository's `download_alimeeting_translation.sh`, which fetches the Microsoft Research `test.txt` file. The raw file is six-column TSV (`recording_id`, original speaker ID, start, end, Chinese text, English translation); the generated JSONL normalizes speaker IDs to `S01...` and emits MOSS timestamp/speaker references. The test file currently hashes to `130d0e33805f9c767463ae8dfce8494d1cee3dd9ce8d7a159203a62a07058e2b`, with 20 sessions and 16,358 utterances. We also fetched the official OpenSLR Test_Ali archive to Lexar (`9555497666` bytes, SHA-256 `42407625b774a44fe34f2f25ca99116f090bec901f5ab81196b89f683a8b7bfa`) and verified a 20-row far-field audio/reference join. This is still not a claim that MOSS's internal release is identical; preserve the upstream attribution and verify redistribution permissions before publishing.

All audio is outside the git checkout on the local laptop/Lexar volume; neither the GPU host nor the `quant` CPU host is used for dataset storage. The current bundle is `/Volumes/Lexar/moss-transcribe-diarize-public-eval/` and includes pinned revisions, SHA-256 file hashes, and generated manifests. The original temporary laptop copy is retained until release verification is complete.

Expected layout on the external benchmark store:

```text
/Volumes/Lexar/moss-transcribe-diarize-public-eval/
  aishell4/validation/{audio,metadata.jsonl}/...
  movies800time/validation/{audio,metadata.jsonl}/...
  movies800/validation/{audio,metadata.jsonl}/...
  googletime/validation/{audio,metadata.jsonl}/...
  alimeeting_references/{diarist_alimeeting_test.txt,diarist_alimeeting_test.jsonl,reference_receipt.json}
  alimeeting_official/{Test_Ali.tar.gz,archive_receipt.json,extracted/Test_Ali/{Test_Ali_far,Test_Ali_near}/...}
  manifests/{aishell4,alimeeting,googletime,movies800,movies800time,googletime_en,movies800_en,movies800time_en}.jsonl
  download_receipt.json
```

Every model comparison must record source revision, selected files, normalization, output text, CER, cpCER, delta_cp, wall time, CPU model/threads, peak RSS, and the parent/QAT checkpoint hash. A MOSS-table number is not considered reproducible until the corresponding audio and reference manifest are present.
