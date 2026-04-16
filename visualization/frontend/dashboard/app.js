(function () {
  const dataset = window.ARENA_REPORT_DATA;

  const elements = {
    errorBanner: document.getElementById("error-banner"),
    metaPill: document.getElementById("meta-pill"),
    sourceNote: document.getElementById("source-note"),
    summaryCards: document.getElementById("summary-cards"),
    leaderboardBody: document.getElementById("leaderboard-body"),
    heatmapHead: document.getElementById("heatmap-head"),
    heatmapBody: document.getElementById("heatmap-body"),
    heatmapCaption: document.getElementById("heatmap-caption"),
    spotlightGrid: document.getElementById("spotlight-grid"),
    taskTableHead: document.getElementById("task-table-head"),
    taskTableBody: document.getElementById("task-table-body"),
    taskTableCaption: document.getElementById("task-table-caption"),
    reportChips: document.getElementById("report-chips"),
    taskTypeChips: document.getElementById("task-type-chips"),
    statusChips: document.getElementById("status-chips"),
    taskSearch: document.getElementById("task-search"),
    heatmapMetric: document.getElementById("heatmap-metric"),
    taskSort: document.getElementById("task-sort"),
  };

  if (!dataset || !Array.isArray(dataset.reports)) {
    showFatal(
      "Dashboard data not found. Run `python backend/scripts/build_dashboard_data.py` to generate `frontend/dashboard/data.js` first."
    );
    return;
  }

  const reports = dataset.reports.slice();
  const taskCatalog = Array.isArray(dataset.taskCatalog) ? dataset.taskCatalog.slice() : [];
  const taskTypeOrder = Object.keys(dataset.taskTypeTotals || {}).sort(
    (left, right) => (dataset.taskTypeTotals[right] || 0) - (dataset.taskTypeTotals[left] || 0)
  );
  const allTaskTypes = taskTypeOrder.length
    ? taskTypeOrder
    : Array.from(new Set(taskCatalog.map((item) => item.taskType))).sort();
  const allStatuses = Array.from(
    new Set([...(dataset.meta?.statuses || []), "PASS", "PARTIAL", "FAIL", "UNKNOWN"])
  ).sort((left, right) => statusRank(left) - statusRank(right));

  const reportContexts = new Map(
    reports.map((report) => [
      report.id,
      {
        report,
        tasksByName: new Map((report.tasks || []).map((task) => [task.taskName, task])),
      },
    ])
  );

  const state = {
    selectedReports: new Set(reports.map((report) => report.id)),
    selectedTaskTypes: new Set(allTaskTypes),
    selectedStatuses: new Set(allStatuses),
    search: "",
    heatmapMetric: elements.heatmapMetric.value,
    taskSort: elements.taskSort.value,
  };

  bindEvents();
  render();

  function bindEvents() {
    elements.taskSearch.addEventListener("input", (event) => {
      state.search = event.target.value.trim().toLowerCase();
      render();
    });

    elements.heatmapMetric.addEventListener("change", (event) => {
      state.heatmapMetric = event.target.value;
      render();
    });

    elements.taskSort.addEventListener("change", (event) => {
      state.taskSort = event.target.value;
      render();
    });

    document.querySelectorAll("[data-filter-action]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.filterAction;
        if (action === "toggle-reports") {
          toggleAll(state.selectedReports, reports.map((report) => report.id));
        } else if (action === "toggle-task-types") {
          toggleAll(state.selectedTaskTypes, allTaskTypes);
        } else if (action === "toggle-statuses") {
          toggleAll(state.selectedStatuses, allStatuses);
        }
        render();
      });
    });
  }

  function toggleAll(targetSet, allItems) {
    if (targetSet.size === allItems.length) {
      targetSet.clear();
      return;
    }
    targetSet.clear();
    allItems.forEach((item) => targetSet.add(item));
  }

  function render() {
    renderMeta();
    renderWarnings();
    renderChips(elements.reportChips, reports, state.selectedReports, "report", (report) => ({
      key: report.id,
      label: report.label,
      title: report.runTimestampFormatted || report.runTimestamp || report.label,
    }));
    renderChips(
      elements.taskTypeChips,
      allTaskTypes,
      state.selectedTaskTypes,
      "task-type",
      (taskType) => ({ key: taskType, label: taskType })
    );
    renderChips(elements.statusChips, allStatuses, state.selectedStatuses, "status", (status) => ({
      key: status,
      label: status,
    }));

    const visibleReports = getVisibleReports();
    const taskRows = buildTaskRows(visibleReports);

    renderSummaryCards(visibleReports);
    renderLeaderboard(visibleReports);
    renderHeatmap(visibleReports);
    renderSpotlights(visibleReports, taskRows);
    renderTaskTable(visibleReports, taskRows);
  }

  function renderMeta() {
    const latestRun = dataset.meta?.latestRunTimestampFormatted || "n/a";
    elements.metaPill.textContent = `${reports.length} reports • latest run ${latestRun}`;
    const gpuLabel = (dataset.meta?.targetGpus || []).join(", ") || "n/a";
    elements.sourceNote.textContent = [
      `Generated ${dataset.meta?.generatedAt || "unknown"}`,
      `Tasks ${dataset.meta?.taskCount || 0}`,
      `GPU ${gpuLabel}`,
      `Workspace scan ${dataset.meta?.includeWorkspaceRuns ? "on" : "off"}`,
      `Scan root ${dataset.meta?.scanRoot || "unknown"}`,
    ].join(" • ");
  }

  function renderWarnings() {
    const warnings = dataset.meta?.warnings || [];
    if (!warnings.length) {
      elements.errorBanner.classList.add("hidden");
      elements.errorBanner.textContent = "";
      return;
    }

    const preview = warnings.slice(0, 4).join(" | ");
    elements.errorBanner.textContent =
      warnings.length > 4 ? `${preview} | +${warnings.length - 4} more` : preview;
    elements.errorBanner.classList.remove("hidden");
  }

  function renderChips(container, items, selectedSet, group, toChip) {
    container.innerHTML = "";
    items.forEach((item) => {
      const chipData = toChip(item);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `chip${selectedSet.has(chipData.key) ? " active" : ""}`;
      button.textContent = chipData.label;
      if (chipData.title) {
        button.title = chipData.title;
      }
      button.addEventListener("click", () => {
        if (selectedSet.has(chipData.key)) {
          selectedSet.delete(chipData.key);
        } else {
          selectedSet.add(chipData.key);
        }
        render();
      });
      button.dataset.group = group;
      button.dataset.key = chipData.key;
      container.appendChild(button);
    });
  }

  function getVisibleReports() {
    return reports.filter((report) => state.selectedReports.has(report.id));
  }

  function buildTaskRows(visibleReports) {
    return taskCatalog
      .map((catalogItem) => {
        const cells = visibleReports.map((report) => {
          const entry = reportContexts.get(report.id);
          return entry?.tasksByName.get(catalogItem.taskName) || null;
        });
        const populatedCells = cells.filter(Boolean);
        const matchesSearch =
          !state.search || catalogItem.taskName.toLowerCase().includes(state.search);
        const matchesTaskType = state.selectedTaskTypes.has(catalogItem.taskType);
        const matchesStatus = populatedCells.some((cell) =>
          state.selectedStatuses.has((cell.status || "UNKNOWN").toUpperCase())
        );

        const maxSpeedup = populatedCells.length
          ? Math.max(...populatedCells.map((cell) => Number(cell.speedup) || 0))
          : -1;
        const avgScore = populatedCells.length
          ? populatedCells.reduce((sum, cell) => sum + (Number(cell.score) || 0), 0) /
            populatedCells.length
          : -1;

        return {
          taskName: catalogItem.taskName,
          taskType: catalogItem.taskType,
          cells,
          populatedCells,
          maxSpeedup,
          avgScore,
          include: matchesSearch && matchesTaskType && matchesStatus,
        };
      })
      .filter((row) => row.include)
      .sort(sortTaskRows);
  }

  function sortTaskRows(left, right) {
    if (state.taskSort === "avg_score_desc") {
      return sortNumberDesc(left.avgScore, right.avgScore) || left.taskName.localeCompare(right.taskName);
    }

    if (state.taskSort === "task_name_asc") {
      return left.taskName.localeCompare(right.taskName);
    }

    if (state.taskSort === "task_type_asc") {
      return left.taskType.localeCompare(right.taskType) || left.taskName.localeCompare(right.taskName);
    }

    return sortNumberDesc(left.maxSpeedup, right.maxSpeedup) || sortNumberDesc(left.avgScore, right.avgScore) || left.taskName.localeCompare(right.taskName);
  }

  function renderSummaryCards(visibleReports) {
    if (!visibleReports.length) {
      elements.summaryCards.innerHTML = `<div class="empty-state">No reports selected.</div>`;
      return;
    }

    const bestScore = maxBy(visibleReports, (report) => report.overall?.total_score || 0);
    const bestMedian = maxBy(visibleReports, (report) => report.overall?.median_speedup || 0);
    const bestCorrect = maxBy(
      visibleReports,
      (report) => report.overall?.correctness_pass_rate || 0
    );
    const longTail = maxBy(visibleReports, (report) => {
      const average = report.overall?.average_speedup || 0;
      const median = report.overall?.median_speedup || 0;
      return median > 0 ? average / median : average;
    });

    const cards = [
      {
        label: "Active reports",
        value: `${visibleReports.length}/${reports.length}`,
        detail: `Current scope • ${dataset.meta?.taskCount || 0} tasks • latest ${
          dataset.meta?.latestRunTimestampFormatted || "n/a"
        }`,
      },
      {
        label: "Highest total score",
        value: formatScore(bestScore.overall?.total_score || 0),
        detail: `${bestScore.label} • avg speedup ${formatSpeedup(bestScore.overall?.average_speedup || 0)}`,
      },
      {
        label: "Best median speedup",
        value: formatSpeedup(bestMedian.overall?.median_speedup || 0),
        detail: `${bestMedian.label} • avg ${formatSpeedup(bestMedian.overall?.average_speedup || 0)}`,
      },
      {
        label: "Best correctness rate",
        value: formatPercent(bestCorrect.overall?.correctness_pass_rate || 0),
        detail: `${bestCorrect.label} • compile ${formatPercent(bestCorrect.overall?.compilation_pass_rate || 0)}`,
      },
      {
        label: "Largest avg / median gap",
        value: formatRatio(
          (longTail.overall?.average_speedup || 0) / Math.max(longTail.overall?.median_speedup || 1, 1e-9)
        ),
        detail: `${longTail.label} • avg ${formatSpeedup(longTail.overall?.average_speedup || 0)} vs median ${formatSpeedup(longTail.overall?.median_speedup || 0)}`,
      },
    ];

    elements.summaryCards.innerHTML = cards
      .map(
        (card) => `
          <article class="summary-card">
            <div class="summary-card-label">${escapeHtml(card.label)}</div>
            <div class="summary-card-value">${escapeHtml(card.value)}</div>
            <div class="summary-card-detail">${escapeHtml(card.detail)}</div>
          </article>
        `
      )
      .join("");
  }

  function renderLeaderboard(visibleReports) {
    if (!visibleReports.length) {
      elements.leaderboardBody.innerHTML =
        '<tr><td colspan="8"><div class="empty-state">No reports selected.</div></td></tr>';
      return;
    }

    const rows = visibleReports
      .slice()
      .sort((left, right) =>
        sortNumberDesc(left.overall?.total_score || 0, right.overall?.total_score || 0)
      )
      .map((report) => {
        const overall = report.overall || {};
        return `
          <tr>
            <td class="leaderboard-report">
              <div class="report-name">${escapeHtml(report.label)}</div>
              <div class="report-meta">${escapeHtml(report.agent || "unknown")} • ${escapeHtml(
          report.runTimestampFormatted || report.runTimestamp || "n/a"
        )}</div>
            </td>
            <td>${formatScore(overall.total_score || 0)}</td>
            <td>${formatSpeedup(overall.average_speedup || 0)}</td>
            <td>${formatSpeedup(overall.median_speedup || 0)}</td>
            <td><span class="metric-pill">${formatPercent(overall.compilation_pass_rate || 0)}</span></td>
            <td><span class="metric-pill">${formatPercent(overall.correctness_pass_rate || 0)}</span></td>
            <td><span class="metric-pill">${formatPercent(overall.speedup_gt_1_rate || 0)}</span></td>
            <td>
              <div class="source-links">
                ${renderSourceLink(report.sourceFiles?.summaryCsv, "CSV")}
                ${renderSourceLink(report.sourceFiles?.breakdownJson, "JSON")}
                ${renderSourceLink(report.sourceFiles?.overallReport, "TXT")}
              </div>
            </td>
          </tr>
        `;
      });

    elements.leaderboardBody.innerHTML = rows.join("");
  }

  function renderHeatmap(visibleReports) {
    const metric = state.heatmapMetric;
    const metricLabel = metricToLabel(metric);
    elements.heatmapCaption.textContent = `${metricLabel} by task type across ${
      visibleReports.length
    } selected reports.`;

    if (!visibleReports.length) {
      elements.heatmapHead.innerHTML = "";
      elements.heatmapBody.innerHTML = `<tr><td><div class="empty-state">No reports selected.</div></td></tr>`;
      return;
    }

    elements.heatmapHead.innerHTML = `
      <tr>
        <th class="heatmap-rowhead">Task type</th>
        ${visibleReports
          .map(
            (report) => `<th>${escapeHtml(report.label)}</th>`
          )
          .join("")}
      </tr>
    `;

    const values = [];
    allTaskTypes.forEach((taskType) => {
      visibleReports.forEach((report) => {
        const taskTypeData = report.taskTypes?.[taskType];
        if (taskTypeData && Number.isFinite(taskTypeData[metric])) {
          values.push(Number(taskTypeData[metric]));
        }
      });
    });

    const minValue = values.length ? Math.min(...values) : 0;
    const maxValue = values.length ? Math.max(...values) : 0;

    elements.heatmapBody.innerHTML = allTaskTypes
      .map((taskType) => {
        const count = dataset.taskTypeTotals?.[taskType] || 0;
        const cells = visibleReports
          .map((report) => {
            const taskTypeData = report.taskTypes?.[taskType];
            if (!taskTypeData) {
              return `<td><div class="heatmap-cell" style="background: rgba(31, 37, 33, 0.06); color: #5f665f;">n/a</div></td>`;
            }

            const value = Number(taskTypeData[metric]) || 0;
            const normalized = normalizeMetric(metric, value, minValue, maxValue);
            const bgColor = heatColor(normalized);
            return `
              <td>
                <div class="heatmap-cell" style="background: ${bgColor}">
                  <div class="heatmap-value">${escapeHtml(formatMetric(metric, value))}</div>
                  <div class="heatmap-subvalue">${escapeHtml(
                    `${taskTypeData.count || 0} tasks`
                  )}</div>
                </div>
              </td>
            `;
          })
          .join("");

        return `
          <tr>
            <th class="heatmap-rowhead">${escapeHtml(taskType)}<div class="report-meta">${count} task records</div></th>
            ${cells}
          </tr>
        `;
      })
      .join("");
  }

  function renderSpotlights(visibleReports, taskRows) {
    if (!taskRows.length) {
      elements.spotlightGrid.innerHTML =
        '<div class="empty-state">No tasks match the current filters.</div>';
      return;
    }

    const spotlightRows = taskRows
      .slice()
      .sort((left, right) => sortNumberDesc(left.maxSpeedup, right.maxSpeedup))
      .slice(0, 6);

    elements.spotlightGrid.innerHTML = spotlightRows
      .map((row) => {
        const winners = bestTaskCells(visibleReports, row);
        const winner = winners[0];
        if (!winner) {
          return "";
        }
        const report = visibleReports.find((item) => item.id === winner.reportId);
        const runnerUp = row.populatedCells
          .filter((cell) => cell !== winner.task)
          .sort((left, right) => sortNumberDesc(left.speedup || 0, right.speedup || 0))[0];
        const runnerText = runnerUp
          ? `runner-up ${formatSpeedup(runnerUp.speedup || 0)}`
          : "single report";

        return `
          <article class="spotlight-card">
            <div class="spotlight-type">${escapeHtml(row.taskType)}</div>
            <div class="spotlight-task">${escapeHtml(row.taskName)}</div>
            <div class="spotlight-score">${escapeHtml(formatSpeedup(winner.task.speedup || 0))}</div>
            <div class="spotlight-caption">
              ${escapeHtml(report?.label || winner.reportId)} • ${escapeHtml(
          winner.task.status || "UNKNOWN"
        )} • score ${escapeHtml(formatScore(winner.task.score || 0))}<br />
              ${escapeHtml(runnerText)}
            </div>
          </article>
        `;
      })
      .join("");
  }

  function renderTaskTable(visibleReports, taskRows) {
    elements.taskTableCaption.textContent = `${taskRows.length} tasks after filters • ${
      visibleReports.length
    } visible reports`;

    if (!visibleReports.length) {
      elements.taskTableHead.innerHTML = "";
      elements.taskTableBody.innerHTML =
        '<tr><td><div class="empty-state">No reports selected.</div></td></tr>';
      return;
    }

    elements.taskTableHead.innerHTML = `
      <tr>
        <th>Task</th>
        <th>Type</th>
        ${visibleReports
          .map(
            (report) =>
              `<th>${escapeHtml(report.label)}<div class="report-meta">${escapeHtml(
                report.runTimestampFormatted || report.runTimestamp || "n/a"
              )}</div></th>`
          )
          .join("")}
      </tr>
    `;

    if (!taskRows.length) {
      elements.taskTableBody.innerHTML =
        `<tr><td colspan="${2 + visibleReports.length}"><div class="empty-state">No tasks match the current filters.</div></td></tr>`;
      return;
    }

    elements.taskTableBody.innerHTML = taskRows
      .map((row) => {
        const bestCells = bestTaskCells(visibleReports, row);
        return `
          <tr>
            <td class="task-name">${escapeHtml(row.taskName)}</td>
            <td>${escapeHtml(row.taskType)}</td>
            ${visibleReports
              .map((report, index) => {
                const task = row.cells[index];
                if (!task) {
                  return `<td class="task-cell"><div class="task-cell-inner"><span class="status-pill status-unknown">N/A</span><div class="task-submetric">missing in report</div></div></td>`;
                }

                const isBest = bestCells.some((cell) => cell.reportId === report.id);
                const status = (task.status || "UNKNOWN").toUpperCase();
                return `
                  <td class="task-cell${isBest ? " best" : ""}">
                    <div class="task-cell-inner" title="${escapeHtml(
                      task.optimizationSummary || "No optimization summary"
                    )}">
                      <span class="status-pill ${statusClass(status)}">${escapeHtml(status)}</span>
                      <div class="task-metric">${escapeHtml(formatSpeedup(task.speedup || 0))}</div>
                      <div class="task-submetric">score ${escapeHtml(formatScore(task.score || 0))}</div>
                    </div>
                  </td>
                `;
              })
              .join("")}
          </tr>
        `;
      })
      .join("");
  }

  function bestTaskCells(visibleReports, row) {
    const cells = row.cells
      .map((task, index) => ({ task, reportId: visibleReports[index]?.id }))
      .filter((item) => item.task && item.reportId);

    if (!cells.length) {
      return [];
    }

    const bestSpeedup = Math.max(...cells.map((item) => Number(item.task.speedup) || 0));
    return cells.filter((item) => (Number(item.task.speedup) || 0) === bestSpeedup);
  }

  function renderSourceLink(path, label) {
    if (!path) {
      return "";
    }
    return `<a class="source-link" href="./${escapeHtml(path)}" target="_blank" rel="noreferrer">${escapeHtml(
      label
    )}</a>`;
  }

  function metricToLabel(metric) {
    const labels = {
      average_speedup: "Average speedup",
      median_speedup: "Median speedup",
      average_score: "Average score",
      correctness_pass_rate: "Correctness pass rate",
      compilation_pass_rate: "Compilation pass rate",
      speedup_gt_1_rate: "Speedup > 1 rate",
    };
    return labels[metric] || metric;
  }

  function formatMetric(metric, value) {
    if (metric.includes("speedup")) {
      return formatSpeedup(value);
    }
    if (metric.includes("rate")) {
      return formatPercent(value);
    }
    return formatScore(value);
  }

  function normalizeMetric(metric, value, minValue, maxValue) {
    if (metric.includes("rate")) {
      return clamp(value / 100, 0, 1);
    }

    const normalizer = metric === "average_score" ? Math.log1p : Math.log1p;
    const low = normalizer(Math.max(minValue, 0));
    const high = normalizer(Math.max(maxValue, 0));
    if (high === low) {
      return 0.5;
    }
    return clamp((normalizer(Math.max(value, 0)) - low) / (high - low), 0, 1);
  }

  function heatColor(normalized) {
    const hue = 20 + normalized * 145;
    const saturation = 58 + normalized * 16;
    const lightness = 92 - normalized * 34;
    return `hsl(${hue} ${saturation}% ${lightness}%)`;
  }

  function showFatal(message) {
    elements.errorBanner.textContent = message;
    elements.errorBanner.classList.remove("hidden");
  }

  function formatSpeedup(value) {
    return `${Number(value || 0).toFixed(2)}x`;
  }

  function formatPercent(value) {
    return `${Number(value || 0).toFixed(1)}%`;
  }

  function formatScore(value) {
    return Number(value || 0).toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 1,
    });
  }

  function formatRatio(value) {
    return `${Number(value || 0).toFixed(2)}x`;
  }

  function sortNumberDesc(left, right) {
    return (right || 0) - (left || 0);
  }

  function maxBy(items, metric) {
    return items.reduce((best, item) => (metric(item) > metric(best) ? item : best), items[0]);
  }

  function escapeHtml(value) {
    const lookup = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return String(value).replace(/[&<>"']/g, (char) => lookup[char] || char);
  }

  function statusClass(status) {
    if (status === "PASS") {
      return "status-pass";
    }
    if (status === "PARTIAL") {
      return "status-partial";
    }
    if (status === "FAIL") {
      return "status-fail";
    }
    return "status-unknown";
  }

  function statusRank(status) {
    if (status === "PASS") {
      return 0;
    }
    if (status === "PARTIAL") {
      return 1;
    }
    if (status === "FAIL") {
      return 2;
    }
    return 3;
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }
})();
