import { ArrowRightOutlined } from "@ant-design/icons";
import styles from "./Welcome.module.css";

export const USAGE =
  "空会话首页：居中标题与一段口径说明，下面挂三条示例问题（点一条直接提问），底部三行是该服务对降级与依据的承诺。";

type Props = {
  samples: string[];
  onAsk: (question: string) => void;
};

export function Welcome({ samples, onAsk }: Props) {
  return (
    <div className={styles.welcome}>
      <div className={styles.head}>
        <h1 className={styles.title}>问一条交通法规问题</h1>
        <p className={styles.desc}>
          回答会先以草稿流式出现，复核完成后整篇换成权威版本；涉及具体地区时，服务会先问按哪里的规定作答，再继续。
        </p>
      </div>

      <div className={styles.samples}>
        {samples.map((sample) => (
          <button key={sample} type="button" className={styles.sample} onClick={() => onAsk(sample)}>
            <span className={styles.sampleText}>{sample}</span>
            <ArrowRightOutlined className={styles.sampleArrow} />
          </button>
        ))}
      </div>

      <ul className={styles.notes}>
        <li>依据不足、复核没跑、服务降级这几种情况，界面会明说，不假装答案经过复核。</li>
        <li>每条结论后面都跟着可点的依据，右栏能翻到条文原文与这一轮的处理轨迹。</li>
        <li>顶部状态条盯的是服务与检索通道的健康；降级不等于不可用，它会照常回答。</li>
      </ul>
    </div>
  );
}
