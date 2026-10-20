# 09. Storage Policy

```text
storage/
├── builtin/
│   └── {property}/
│       └── models/
│           └── {model_id}/{version}/
│               ├── metadata.json
│               ├── fold_1/
│               ├── fold_2/
│               ├── fold_3/
│               ├── fold_4/
│               └── fold_5/
└── users/
    └── {user_id}/
        ├── datasets/
        │   └── {dataset_id}/{version}/
        ├── representations/
        │   └── {representation_id}/
        └── models/
            └── {model_id}/{version}/
                ├── metadata.json
                ├── fold_1/
                ├── fold_2/
                ├── fold_3/
                ├── fold_4/
                └── fold_5/
```

각 fold directory에는 model.joblib 및 필요 시 split/descriptor/GA metadata를 둔다.

사용자/Admin이 arbitrary filesystem path를 지정하지 않는다.
Storage Service가 canonical object key를 생성한다.

DB에는 object key, checksum, ownership, artifact type, fold_no 등을 저장한다.
