import { Button, Tooltip } from "antd";
import { SettingOutlined } from "@ant-design/icons";
import type { ApiMode } from "../config";
import type { HealthState } from "../state/health";
import { StatusBar } from "./StatusBar";
import styles from "./ChatHeader.module.css";

export const USAGE =
  "对话页页眉：左数据来源标签、中会话标题、右健康状态条与设置入口；探活详情归 StatusBar 的 Popover。";

type Props = {
  title: string;
  mode: ApiMode;
  health: HealthState;
  onOpenSettings: () => void;
};

export function ChatHeader({ title, mode, health, onOpenSettings }: Props) {
  return (
    <header className={styles.header}>
      <div className={styles.side}>
        <span className={styles.sourceTag}>{mode === "live" ? "直连服务" : "示例数据"}</span>
      </div>
      <div className={styles.title} title={title}>
        {title}
      </div>
      <div className={`${styles.side} ${styles.sideRight}`}>
        <StatusBar health={health} mode={mode} />
        <Tooltip title="设置">
          <Button type="text" shape="circle" icon={<SettingOutlined />} onClick={onOpenSettings} />
        </Tooltip>
      </div>
    </header>
  );
}
