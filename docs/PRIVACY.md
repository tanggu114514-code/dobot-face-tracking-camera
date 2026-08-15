# Privacy

This project processes biometric face data. Use it only with informed consent
and in compliance with applicable rules and laws.

## Local data

Enrollment creates files under:

```text
data/face_dataset/
data/targets/
data/label_mapping.json
data/face_rec_model.yml
```

These paths are ignored by Git. Do not remove those ignore rules, publish the
files, attach them to issues, or include them in demonstration archives.

## Recommended practice

- Explain the purpose, storage and deletion procedure before enrollment.
- Collect only the minimum number of images needed for the demonstration.
- Keep enrollment data on the local test computer.
- Do not use the system for covert identification or surveillance.
- Delete generated files when a participant withdraws consent.
- Review every archive with `python -m unittest discover -s tests -v` before release.

The ONNX face models are general pretrained models and contain no identity
enrollment from this project.
