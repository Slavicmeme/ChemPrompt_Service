# 07. Aya Representation Service

Aya (`CohereLabs/aya-expanse-8b`)는 사용자 Agent가 아니다.

역할:
ChemPrompt prompt -> molecular representation/embedding.

GPU/H100 server에서 별도 서비스로 배포 가능하게 한다.

sklearn model은 Representation Service의 결과를 소비하며 GPU server와 deployment가 결합되지 않는다.

Representation cache identity:
- dataset version/checksum
- Aya model/version
- prompt version
- descriptor mode
- fold/GA descriptor config가 representation을 변화시키는 경우 해당 config
- preprocessing version
