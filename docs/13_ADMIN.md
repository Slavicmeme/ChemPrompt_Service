# 13. Admin

관리자 기능:
- model upload
- model validation
- model version 관리
- serving 전환
- user custom model 검토/승인
- dataset/training/model metadata 확인

Admin workstation은 sklearn 작업을 CPU에서도 수행 가능해야 한다.

Aya representation이 필요한 경우 GPU server API를 호출하고 결과를 받아 sklearn 학습한다.

Admin이라고 해도 storage path를 직접 입력하지 않는다.
