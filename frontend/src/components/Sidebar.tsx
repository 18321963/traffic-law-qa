import { Button } from "antd";
import { CloseOutlined, PlusOutlined, SettingOutlined } from "@ant-design/icons";
import type { ApiMode } from "../config";
import type { Conversation } from "../state/conversation";
import { BrandMark } from "./BrandMark";
import styles from "./Sidebar.module.css";

export const USAGE =
  "左栏：品牌区、新会话、本机会话历史（时间 + 标题，悬停可清掉）；底部是数据来源与设置入口。";

type Props = {
  conversations: Conversation[];
  activeId: string;
  mode: ApiMode;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRemove: (id: string) => void;
  onOpenSettings: () => void;
};

function stamp(ms: number): string {
  const date = new Date(ms);
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  const hour = `${date.getHours()}`.padStart(2, "0");
  const minute = `${date.getMinutes()}`.padStart(2, "0");
  return `${hour}:${minute} ${month}/${day}`;
}

export function Sidebar({ conversations, activeId, mode, onSelect, onCreate, onRemove, onOpenSettings }: Props) {
  return (
    <nav className={styles.sidebar}>
      <div className={styles.logo}>
        <BrandMark size={30} />
        <span className={styles.logoTitle}>交通法规问答</span>
      </div>

      <button type="button" className={styles.newChat} onClick={onCreate}>
        <PlusOutlined className={styles.newChatIcon} />
        <span>新会话</span>
      </button>

      <div className={styles.divider} />

      <div className={styles.sectionLabel}>历史会话</div>
      <div className={`${styles.list} scrollbar`}>
        {conversations.map((conv) => {
          const active = conv.id === activeId;
          return (
            <div key={conv.id} className={`${styles.item} ${active ? styles.itemActive : ""}`}>
              <button type="button" className={styles.itemMain} onClick={() => onSelect(conv.id)}>
                <span className={styles.itemTime}>
                  {stamp(conv.createdAt)}
                  <span className={styles.itemMeta}>
                    {conv.turns.length} 轮{conv.sessionId ? " · 带记忆" : ""}
                  </span>
                </span>
                <span className={styles.itemTitle}>{conv.title}</span>
              </button>
              {conversations.length > 1 ? (
                <button
                  type="button"
                  className={styles.itemRemove}
                  title="清掉这条会话的本地记录"
                  onClick={() => onRemove(conv.id)}
                >
                  <CloseOutlined />
                </button>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className={styles.foot}>
        <p className={styles.footNote}>会话只是本机的记录，不会同步到服务；带 session_id 的那条才有跨轮记忆。</p>
        <div className={styles.footRow}>
          <span className={styles.modeChip}>{mode === "live" ? "直连服务" : "示例数据"}</span>
          <Button type="text" size="small" icon={<SettingOutlined />} onClick={onOpenSettings}>
            设置
          </Button>
        </div>
      </div>
    </nav>
  );
}
