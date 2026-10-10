import type { Conversation } from "../state/conversation";

type Props = {
  conversations: Conversation[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRemove: (id: string) => void;
};

function stamp(ms: number): string {
  const date = new Date(ms);
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  const hour = `${date.getHours()}`.padStart(2, "0");
  const minute = `${date.getMinutes()}`.padStart(2, "0");
  return `${month}-${day} ${hour}:${minute}`;
}

export function Sidebar({ conversations, activeId, onSelect, onCreate, onRemove }: Props) {
  return (
    <nav className="sidebar">
      <div className="sidebar__head">
        <h2>会话</h2>
        <button type="button" className="btn btn--quiet" onClick={onCreate}>
          新会话
        </button>
      </div>
      <ul className="sidebar__list">
        {conversations.map((conv) => (
          <li key={conv.id} className={conv.id === activeId ? "is-active" : ""}>
            <button type="button" className="sidebar__item" onClick={() => onSelect(conv.id)}>
              <span className="sidebar__title">{conv.title}</span>
              <span className="sidebar__meta">
                {conv.turns.length} 轮 · {stamp(conv.createdAt)}
                {conv.sessionId ? " · 带记忆" : ""}
              </span>
            </button>
            {conversations.length > 1 ? (
              <button
                type="button"
                className="sidebar__remove"
                title="清掉这条会话的本地记录"
                onClick={() => onRemove(conv.id)}
              >
                ×
              </button>
            ) : null}
          </li>
        ))}
      </ul>
      <p className="sidebar__note">会话只是本机的记录，不会同步到服务；带 session_id 的那条才有跨轮记忆。</p>
    </nav>
  );
}
