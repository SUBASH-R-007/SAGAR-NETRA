export default function FilterBar({
  surveys,
  survey,
  onSurvey,
  classes,
  cls,
  onCls,
  minConf,
  onMinConf,
  shown,
  total,
}) {
  return (
    <div className="filterbar">
      <label className="ctl">
        <span className="ctl-label">Survey</span>
        <select value={survey} onChange={(e) => onSurvey(e.target.value)}>
          {surveys.length === 0 && <option value="">— no surveys yet —</option>}
          {surveys.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
              {s.n_contacts != null ? ` (${s.n_contacts})` : ''}
            </option>
          ))}
        </select>
      </label>

      <label className="ctl">
        <span className="ctl-label">Class</span>
        <select value={cls} onChange={(e) => onCls(e.target.value)}>
          <option value="all">all classes</option>
          {classes.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>

      <label className="ctl slider">
        <span className="ctl-label">Min confidence</span>
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={minConf}
          onChange={(e) => onMinConf(Number(e.target.value))}
        />
        <span className="mono ctl-value">{minConf}%</span>
      </label>

      <span className="filter-count mono">
        {shown}/{total} contacts
      </span>
    </div>
  )
}
