# Redis 使用手册

## Redis 的用途

Redis 是一个基于内存的 Key-Value 数据库，读写速度极快。常见用途包括：缓存、分布式锁、计数器、排行榜、简单消息队列（List / Stream）。
在我们团队，Redis 集群的内部代号是「青鸟」（Bluebird），主要用于缓存用户会话（Session）和接口限流。
团队规范要求：所有 key 必须以 `demo:` 作为前缀，并且必须设置过期时间。

## 接口限流方案

团队的接口限流统一基于 Redis 的 Sorted Set（ZSET）实现滑动窗口算法：每次请求以时间戳作为 score 写入 ZSET，并删除窗口之外的旧记录。
默认限额为每个用户每分钟 120 次请求，超过限额时网关返回 HTTP 429，错误码为 `RATE_LIMITED`。
限流相关的 key 统一使用 `demo:ratelimit:{user_id}` 格式。

## Redis 故障排查

线上 Redis 变慢时，优先排查三类问题：大 key、热 key、慢命令。
团队禁止写入超过 10MB 的单个 value；禁止在生产环境执行 `KEYS *`，遍历 key 必须使用 `SCAN`。
慢命令通过 `SLOWLOG GET` 查看，执行时间超过 10 毫秒的命令会被记录并触发告警。
