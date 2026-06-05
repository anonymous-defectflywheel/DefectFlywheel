# 方案 A：提高 inotify 上限（解决 VS Code ENOSPC）

## 1. 查看当前上限

```bash
cat /proc/sys/fs/inotify/max_user_watches
```

当前常见为 65536 或 52428。

## 2. 应用新上限（需在本机终端执行，会提示输入 sudo 密码）

**方式一：直接写系统配置并生效（推荐）**

```bash
echo "fs.inotify.max_user_watches=524288" | sudo tee /etc/sysctl.d/99-inotify-watches.conf
sudo sysctl --system
```

**方式二：若系统没有 /etc/sysctl.d/，则追加到 sysctl.conf**

```bash
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 3. 验证

```bash
cat /proc/sys/fs/inotify/max_user_watches
```

应显示 `524288`。

## 4. 重启 VS Code / Cursor

关闭后重新打开工作区即可。

---

说明：本目录下的 `set_inotify_watches.conf` 仅为内容参考，实际生效需将上述命令在**本机终端**中执行（需 sudo）。
