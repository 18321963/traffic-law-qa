import { Divider, Drawer, Input, InputNumber, Segmented, Select } from "antd";
import { HEALTH_SCENARIOS, SCENARIOS } from "../api/mock";
import type { Settings } from "../config";
import styles from "./SettingsPanel.module.css";

export const USAGE =
  "设置抽屉：数据来源（示例数据 / 直连服务）、服务地址与 key、答案模式与 top_k；示例数据模式下可挑问答场景与状态条场景演练各条路径。";

type Props = {
  open: boolean;
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  onClose: () => void;
};

export function SettingsPanel({ open, settings, onChange, onClose }: Props) {
  const scenario = SCENARIOS.find((row) => row.id === settings.scenario) ?? SCENARIOS[0];
  const healthScenario = HEALTH_SCENARIOS.find((row) => row.id === settings.healthScenario) ?? HEALTH_SCENARIOS[0];

  return (
    <Drawer title="设置" placement="right" width={420} open={open} onClose={onClose} className={styles.drawer}>
      <div className={styles.section}>
        <div className={styles.sectionTitle}>数据来源</div>
        <Segmented
          block
          value={settings.mode}
          onChange={(value) => onChange({ mode: value as Settings["mode"] })}
          options={[
            { label: "示例数据", value: "mock" },
            { label: "直连服务", value: "live" },
          ]}
        />
        <p className={styles.hint}>
          {settings.mode === "mock"
            ? "用内置夹具回答：覆盖流式、澄清、降级、限流、未就绪等全部帧与状态码，不出网。"
            : "直接请求真实服务：地址留空走同源（开发服务器代理 / 部署后 nginx 反代），也可以填 http://127.0.0.1:8001。"}
        </p>
      </div>

      <Divider className={styles.divider} />

      <div className={styles.section}>
        <div className={styles.sectionTitle}>服务地址</div>
        <Input
          value={settings.base}
          placeholder="留空 = 同源"
          onChange={(event) => onChange({ base: event.target.value.trim() })}
          allowClear
        />
        <div className={styles.sectionTitle + " " + styles.gap}>X-API-Key</div>
        <Input.Password
          value={settings.apiKey}
          placeholder="服务开了鉴权时填"
          onChange={(event) => onChange({ apiKey: event.target.value.trim() })}
        />
        <p className={styles.hint}>key 只存在这台浏览器的 localStorage 里，请求时放进 X-API-Key 头。</p>
      </div>

      <Divider className={styles.divider} />

      <div className={styles.section}>
        <div className={styles.sectionTitle}>问答</div>
        <Segmented
          block
          value={settings.answerMode}
          onChange={(value) => onChange({ answerMode: value as Settings["answerMode"] })}
          options={[
            { label: "流式", value: "stream" },
            { label: "一次性", value: "once" },
          ]}
        />
        <p className={styles.hint}>
          {settings.answerMode === "stream"
            ? "POST /qa/stream：草稿增量出现，复核完成后整篇换成权威答复。"
            : "POST /qa：等整篇答复一次性返回，中途没有草稿。"}
        </p>
        <div className={styles.sectionTitle + " " + styles.gap}>top_k</div>
        <InputNumber
          min={1}
          max={20}
          value={settings.topK}
          placeholder="默认"
          style={{ width: "100%" }}
          onChange={(value) => onChange({ topK: value && value > 0 ? value : null })}
        />
        <p className={styles.hint}>每轮检索带回的条文数；留空用服务默认。</p>
      </div>

      {settings.mode === "mock" ? (
        <>
          <Divider className={styles.divider} />
          <div className={styles.section}>
            <div className={styles.sectionTitle}>演练场景（仅示例数据）</div>
            <Select
              value={settings.scenario}
              style={{ width: "100%" }}
              onChange={(value) => onChange({ scenario: value })}
              options={SCENARIOS.map((row) => ({ label: row.label, value: row.id }))}
            />
            <p className={styles.hint}>{scenario.hint}</p>
            <div className={styles.sectionTitle + " " + styles.gap}>状态条场景</div>
            <Select
              value={settings.healthScenario}
              style={{ width: "100%" }}
              onChange={(value) => onChange({ healthScenario: value })}
              options={HEALTH_SCENARIOS.map((row) => ({ label: row.label, value: row.id }))}
            />
            <p className={styles.hint}>{healthScenario.hint}</p>
          </div>
        </>
      ) : null}
    </Drawer>
  );
}
