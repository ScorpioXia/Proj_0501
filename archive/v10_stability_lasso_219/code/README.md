# v7 executable snapshot

Run from the project root with the `nnUNet-master` environment:

```powershell
C:\ProgramData\anaconda3\envs\nnUNet-master\python.exe archive\v7\code\run_experiment.py
```

The snapshot is for provenance. It expects the project data paths used by the
entry point and writes to the live v7 result directory unless its output path
is changed.
