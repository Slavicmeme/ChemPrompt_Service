-- 009. Property display info for researchers: Korean names and per-target explanations

ALTER TABLE properties ADD COLUMN IF NOT EXISTS name_ko     VARCHAR(200);
ALTER TABLE properties ADD COLUMN IF NOT EXISTS target_info JSONB NOT NULL DEFAULT '{}'::jsonb;

-- target_info: { "<target column>": { "name_ko", "description", "positive", "negative" } }
--   regression: positive/negative omitted; classification: meaning of label 1 / 0

UPDATE properties SET
    name_ko = '수화 자유에너지',
    unit = 'kcal/mol',
    description = '분자가 기체 상태에서 물에 녹아 들어갈 때의 자유에너지 변화(실험값). 음수일수록 물과 잘 어울리는(친수성) 분자임',
    target_info = '{"label": {"name_ko": "수화 자유에너지", "description": "음수일수록 물과 잘 어울림(친수성), 양수일수록 물을 싫어함(소수성)"}}'
WHERE code = 'FreeSolv';

UPDATE properties SET
    name_ko = '수용해도',
    description = '물에 녹는 정도, 포화 용해도의 로그값 log(mol/L). 클수록 물에 잘 녹음',
    target_info = '{"label": {"name_ko": "수용해도", "description": "클수록 물에 잘 녹음 (예: 0 = 1 mol/L, -2 = 0.01 mol/L)"}}'
WHERE code = 'ESOL';

UPDATE properties SET
    name_ko = '친유성 (지질 분배계수)',
    description = 'pH 7.4에서 옥탄올/물 사이의 분배계수 로그값(logD). 클수록 지방(세포막) 쪽으로 잘 분배됨',
    target_info = '{"label": {"name_ko": "친유성 (logD)", "description": "클수록 지방(세포막) 쪽으로 잘 분배됨, 작을수록 물 쪽에 머묾"}}'
WHERE code = 'Lipo';

UPDATE properties SET
    name_ko = '임상 독성',
    description = '약물 후보의 FDA 승인 여부와 임상시험 독성 실패 여부, 두 가지 예/아니오(1/0) 분류',
    target_info = '{"FDA_APPROVED": {"name_ko": "FDA 승인 여부", "description": "미국 FDA에서 승인된 약물인지 여부", "positive": "FDA 승인 약물", "negative": "미승인"},
                    "CT_TOX": {"name_ko": "임상시험 독성", "description": "독성 때문에 임상시험에서 실패했는지 여부", "positive": "독성으로 임상시험 실패", "negative": "독성으로 인한 실패 없음"}}'
WHERE code = 'ClinTox';
