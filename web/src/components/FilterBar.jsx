export default function FilterBar({
  surveys,
  survey,
  onDeleteSurvey,
  onSurvey,
  classes,
  cls,
  onCls,
  minConf,
  onMinConf,
  review,
  onReview,
  shown,
  total,
  // Overview and Recovery are driven by the selected survey but showed no way
  // to change it - a user who landed on either was stuck with whatever survey
  // the console had picked. They get the selector without the contact filters,
  // because Overview deliberately reports unfiltered totals and a filter row
  // above unfiltered numbers is a worse lie than no filter row at all.
  showFilters = true,
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
        {survey && onDeleteSurvey && (
          <button
            type="button"
            className="btn tiny danger"
            title="Delete this survey and all its contacts"
            onClick={onDeleteSurvey}
          >
            delete
          </button>
        )}
      </label>

      {showFilters && (
      <>
      <label className="ctl">
        <span className="ctl-label">Review</span>
        <select value={review} onChange={(e) => onReview(e.target.value)}
          aria-label="Filter by review status">
          <option value="all">all statuses</option>
          <option value="pending">pending</option>
          <option value="confirmed">confirmed</option>
          <option value="rejected">rejected</option>
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
      </>
      )}

      <span className="filter-count mono">
        {showFilters ? `${shown}/${total} contacts` : `${total} contacts`}
      </span>
    </div>
  )
}
