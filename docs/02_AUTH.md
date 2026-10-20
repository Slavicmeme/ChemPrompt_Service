# 02. Auth

- username/password
- password hash: Argon2id 또는 동급
- ownership은 내부 users.id 사용
- roles: USER, ADMIN
- Admin endpoint는 서버에서 role 검증
- secret/API key는 browser에 전달하지 않는다.
