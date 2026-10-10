import { Button, Popconfirm } from "antd";
import { CloseOutlined, PlusOutlined, SettingOutlined } from "@ant-design/icons";
import type { ApiMode } from "../config";
import type { Conversation } from "../state/conversation";
import { BrandMark } from "./BrandMark";
import styles from "./Sidebar.module.css";

export const USAGE =
  "左栏：品牌区、新会话、本机会话历史（时间 + 标题，悬停可删除 —— 本机记录与服务端会话记忆一起清）；底部是数据来源与设置入口。";

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
          const streaming = conv.turns.at(-1)?.status === "streaming";
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
                <Popconfirm
                  title="删除这条会话？"
                  description="本机记录与服务端的会话记忆会一起删掉，删了恢复不了；已挂的材料不受影响。"
                  okText="删除"
                  cancelText="再想想"
                  okButtonProps={{ danger: true }}
                  disabled={streaming}
                  onConfirm={() => onRemove(conv.id)}
                >
                  <button
                    type="button"
                    className={styles.itemRemove}
                    title={streaming ? "这条会话正在作答，等它结束再删" : "删除这条会话（本机记录与服务端记忆一起删）"}
                    disabled={streaming}
                  >
                    <CloseOutlined />
                  </button>
                </Popconfirm>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className={styles.foot}>
        <p className={styles.footNote}>会话 = 本机记录 + 服务端记忆（问过才有）；删会话两边一起清，材料不受影响。</p>
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
