# script

目标：生成可复现、fail-closed 的 runner/parser/回归或数据采集脚本。

仅在相关实验任务读取对应 EXPERIMENTS 条目和协议。Runner 只执行并记录原始事实；parser 是唯一 verdict authority。

必须记录：commit/worktree、binary/model/script/hash、argv/env、硬件/OS/cgroup、case 顺序、stdout/stderr/exit、残留进程和 artifact。缺 case、重复、顺序/身份/字段错误、UNVERIFIED 或伪造 PASS 必须非零退出。

fixture 优先来自真实 producer 样例或与 producer schema 绑定；不能只用自造字符串自证。先做 py/shell syntax、正负 fixture 和 dry-run；真实模型由用户执行。

若修改 parser/Harness，运行实际对应 Gate。输出文件、参数、PASS/FAIL 规则、短测、用户命令和结果提取方式。
