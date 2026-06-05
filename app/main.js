const FEATURE_LABELS = {
  elo_diff: ["Elo 强度", "战队长期强弱，按赛前 Elo 差标准化"],
  team_recent_diff: ["近期状态", "战队最近 10 局平滑胜率"],
  side_profile_diff: ["红蓝方适配", "战队在当前红/蓝方的历史平滑胜率"],
  player_form_diff: ["选手状态", "当前阵容选手的赛前综合表现均值"],
  player_champion_diff: ["选手英雄熟练度", "选手使用所选英雄的历史平滑胜率"],
  team_champion_diff: ["队伍英雄体系", "战队使用所选英雄组合的历史平滑胜率"],
  champion_meta_diff: ["英雄版本强度", "英雄在对应位置的全局历史平滑胜率"],
  patch_experience_diff: ["版本经验", "队伍和选手在当前版本的历史出场经验"],
};

const ROLE_LABELS = {
  top: "TOP",
  jng: "JNG",
  mid: "MID",
  bot: "BOT",
  sup: "SUP",
};

let DATA;
let TEAMS = [];

const $ = (id) => document.getElementById(id);
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const num = (value, digits = 3) => Number(value).toFixed(digits);
const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, x))));

function seriesProbability(p, format) {
  const need = format === "bo5" ? 3 : format === "bo3" ? 2 : 1;
  const games = format === "bo5" ? 5 : format === "bo3" ? 3 : 1;
  let total = 0;
  for (let wins = need; wins <= games; wins += 1) {
    total += choose(games, wins) * p ** wins * (1 - p) ** (games - wins);
  }
  return total;
}

function choose(n, k) {
  let result = 1;
  for (let i = 1; i <= k; i += 1) result = (result * (n - k + i)) / i;
  return result;
}

function teamById(id) {
  return TEAMS.find((team) => team.id === id);
}

function findRecord(list, champion, fallback = 0.5) {
  const hit = (list || []).find((item) => item.champion === champion);
  return hit ? hit.rate : fallback;
}

function rolePlayer(team, role) {
  return (team.roster || []).find((player) => player.position === role);
}

function roleMeta(role, champion) {
  return findRecord(DATA.champions_by_role[role], champion, 0.5);
}

function selectedPicks(side) {
  return Object.fromEntries(DATA.roles.map((role) => [role, $(`${side}-${role}`).value]));
}

function sideValues(team, side, picks) {
  const sideRecord = side === "Blue" ? team.blue_side : team.red_side;
  const playerForm = [];
  const playerChampion = [];
  const teamChampion = [];
  const championMeta = [];

  DATA.roles.forEach((role) => {
    const champion = picks[role];
    const player = rolePlayer(team, role);
    playerForm.push(player?.form?.rate ?? 0.5);
    playerChampion.push(findRecord(player?.champions, champion, 0.5));
    teamChampion.push(findRecord(team.team_champions, champion, 0.5));
    championMeta.push(roleMeta(role, champion));
  });

  return {
    recent: team.recent_rate,
    side: sideRecord?.rate ?? 0.5,
    player_form: average(playerForm),
    player_champ: average(playerChampion),
    team_champ: average(teamChampion),
    champ_meta: average(championMeta),
    patch_exp: team.current_patch_experience ?? 0,
  };
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function prediction() {
  const blue = teamById($("blueTeam").value);
  const red = teamById($("redTeam").value);
  const bluePicks = selectedPicks("blue");
  const redPicks = selectedPicks("red");
  const blueValues = sideValues(blue, "Blue", bluePicks);
  const redValues = sideValues(red, "Red", redPicks);

  const raw = {
    elo_diff: (blue.elo - red.elo) / 400,
    team_recent_diff: blueValues.recent - redValues.recent,
    side_profile_diff: blueValues.side - redValues.side,
    player_form_diff: blueValues.player_form - redValues.player_form,
    player_champion_diff: blueValues.player_champ - redValues.player_champ,
    team_champion_diff: blueValues.team_champ - redValues.team_champ,
    champion_meta_diff: blueValues.champ_meta - redValues.champ_meta,
    patch_experience_diff: blueValues.patch_exp - redValues.patch_exp,
  };

  const model = DATA.model;
  let score = model.intercept;
  const contributions = model.features.map((feature) => {
    const z = (raw[feature] - model.scaler.mean[feature]) / model.scaler.std[feature];
    const contribution = model.weights[feature] * z;
    score += contribution;
    return { feature, raw: raw[feature], z, weight: model.weights[feature], contribution };
  });

  const pBlue = sigmoid(score);
  const seriesP = seriesProbability(pBlue, $("seriesFormat").value);
  return { blue, red, blueValues, redValues, raw, pBlue, seriesP, contributions };
}

function renderPrediction() {
  const result = prediction();
  const blueFavored = result.seriesP >= 0.5;
  const format = $("seriesFormat").value.toUpperCase();
  $("probability").textContent = pct(result.seriesP, 1);
  $("winnerLabel").textContent = `${blueFavored ? result.blue.name : result.red.name} ${format}`;
  $("winnerLabel").className = `verdict__side ${blueFavored ? "blue-text" : "red-text"}`;

  $("featureTable").innerHTML = DATA.model.features
    .map((feature) => {
      const blueValue = sideMetric(result.blueValues, feature, result.blue);
      const redValue = sideMetric(result.redValues, feature, result.red);
      return `<tr>
        <td>${FEATURE_LABELS[feature][0]}</td>
        <td>${formatFeatureValue(feature, blueValue)}</td>
        <td>${formatFeatureValue(feature, redValue)}</td>
        <td class="${result.raw[feature] >= 0 ? "blue-text" : "red-text"}">${formatFeatureValue(feature, result.raw[feature], true)}</td>
      </tr>`;
    })
    .join("");

  const maxContribution = Math.max(...result.contributions.map((item) => Math.abs(item.contribution)), 0.01);
  $("contributionList").innerHTML = result.contributions
    .slice()
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .map((item) => {
      const width = Math.min(50, (Math.abs(item.contribution) / maxContribution) * 50);
      const negative = item.contribution < 0;
      return `<div class="contribution">
        <div class="contribution__row">
          <strong>${FEATURE_LABELS[item.feature][0]}</strong>
          <span class="${negative ? "red-text" : "blue-text"}">${item.contribution >= 0 ? "+" : ""}${num(item.contribution, 3)}</span>
        </div>
        <div class="bar"><span class="${negative ? "negative" : ""}" style="width:${width}%"></span></div>
        <div class="muted">${FEATURE_LABELS[item.feature][1]}</div>
      </div>`;
    })
    .join("");

  renderSnapshots(result.blue, result.red);
  renderProfiles(result.blue, result.red);
}

function sideMetric(values, feature, team) {
  if (feature === "elo_diff") return team.elo / 400;
  if (feature === "team_recent_diff") return values.recent;
  if (feature === "side_profile_diff") return values.side;
  if (feature === "player_form_diff") return values.player_form;
  if (feature === "player_champion_diff") return values.player_champ;
  if (feature === "team_champion_diff") return values.team_champ;
  if (feature === "champion_meta_diff") return values.champ_meta;
  if (feature === "patch_experience_diff") return values.patch_exp;
  return 0;
}

function formatFeatureValue(feature, value, signed = false) {
  const prefix = signed && value > 0 ? "+" : "";
  if (feature === "elo_diff") return `${prefix}${num(value * 400, 0)}`;
  if (feature === "patch_experience_diff") return `${prefix}${num(value, 2)}`;
  if (signed) return `${prefix}${pct(value, 1)}`;
  return pct(value, 1);
}

function renderSnapshots(blue, red) {
  $("teamSnapshot").innerHTML = [blue, red]
    .map(
      (team, index) => `<div class="team-card">
        <div class="team-card__top">
          <div>
            <div class="tag">${index === 0 ? "蓝方" : "红方"}</div>
            <div class="team-card__name">${team.name}</div>
          </div>
          <strong>${Math.round(team.elo)}</strong>
        </div>
        <div class="muted">${team.league} · 近况 ${pct(team.recent_rate)} · 总胜率 ${pct(team.overall.rate)}</div>
      </div>`
    )
    .join("");
}

function renderProfiles(blue, red) {
  $("blueProfile").innerHTML = profileHtml(blue, "Blue");
  $("redProfile").innerHTML = profileHtml(red, "Red");
}

function profileHtml(team, side) {
  const sideRate = side === "Blue" ? team.blue_side : team.red_side;
  const champs = (team.team_champions || [])
    .slice(0, 10)
    .map((champ) => `<span class="champion-pill">${champ.champion} · ${champ.games}局 · ${pct(champ.rate)}</span>`)
    .join("");
  const roster = (team.roster || [])
    .map(
      (player) => `<div class="roster-row">
        <span class="tag">${ROLE_LABELS[player.position]}</span>
        <strong>${player.playername}</strong>
        <span>${pct(player.form.rate)}</span>
      </div>`
    )
    .join("");
  return `<article class="team-card">
      <div class="team-card__top">
        <div>
          <div class="team-card__name">${team.name}</div>
          <div class="muted">${team.league} · last seen ${team.last_seen.slice(0, 10)} · patch ${team.patch}</div>
        </div>
        <span class="tag">Elo ${Math.round(team.elo)}</span>
      </div>
      <div class="roster">${roster}</div>
      <h3>当前边胜率</h3>
      <p>${side} side: ${sideRate.games} 局 · ${pct(sideRate.rate)}</p>
      <h3>队伍英雄体系</h3>
      <div class="champion-list">${champs || '<span class="muted">样本不足</span>'}</div>
    </article>`;
}

function hydrateTeamSelects() {
  TEAMS = DATA.teams
    .slice()
    .sort((a, b) => b.elo - a.elo || b.overall.games - a.overall.games);
  const options = TEAMS.map(
    (team) => `<option value="${team.id}">${team.name} · ${team.league} · Elo ${Math.round(team.elo)}</option>`
  ).join("");
  $("blueTeam").innerHTML = options;
  $("redTeam").innerHTML = options;
  $("blueTeam").value = TEAMS[0]?.id;
  $("redTeam").value = TEAMS.find((team) => team.id !== $("blueTeam").value)?.id;
}

function renderDraftControls() {
  $("draftGrid").innerHTML = [
    `<div class="role"></div><strong>蓝方英雄</strong><strong>红方英雄</strong>`,
    ...DATA.roles.map((role) => {
      const options = championOptions(role);
      return `<div class="role">${ROLE_LABELS[role]}</div>
        <select id="blue-${role}">${options}</select>
        <select id="red-${role}">${options}</select>`;
    }),
  ].join("");
  DATA.roles.forEach((role) => {
    $(`blue-${role}`).addEventListener("change", renderPrediction);
    $(`red-${role}`).addEventListener("change", renderPrediction);
  });
}

function championOptions(role) {
  return (DATA.champions_by_role[role] || [])
    .map((champ) => `<option value="${escapeHtml(champ.champion)}">${champ.champion} · ${champ.games}局 · ${pct(champ.rate)}</option>`)
    .join("");
}

function setDraftFromRosters() {
  const blue = teamById($("blueTeam").value);
  const red = teamById($("redTeam").value);
  DATA.roles.forEach((role) => {
    setRoleChampion("blue", role, rolePlayer(blue, role)?.champion);
    setRoleChampion("red", role, rolePlayer(red, role)?.champion);
  });
  renderPrediction();
}

function setRoleChampion(side, role, champion) {
  const select = $(`${side}-${role}`);
  if (!champion) return;
  if (![...select.options].some((option) => option.value === champion)) {
    select.insertAdjacentHTML("afterbegin", `<option value="${escapeHtml(champion)}">${champion} · 最近使用</option>`);
  }
  select.value = champion;
}

function renderModelPanel() {
  const validation = DATA.model.validation;
  $("metricCards").innerHTML = [
    ["验证集", `${validation.games.toLocaleString()} 局`],
    ["Accuracy", pct(validation.accuracy, 2)],
    ["Log loss", num(validation.log_loss, 3)],
    ["Brier / ECE", `${num(validation.brier, 3)} / ${num(validation.ece, 3)}`],
  ]
    .map(([label, value]) => `<div class="metric"><div class="eyebrow">${label}</div><div class="metric__value">${value}</div></div>`)
    .join("");

  $("ablationTable").innerHTML = DATA.model.ablations
    .map(
      (row) => `<tr>
        <td>${row.name}</td>
        <td>${pct(row.accuracy, 2)}</td>
        <td>${num(row.log_loss, 3)}</td>
        <td>${num(row.brier, 3)}</td>
        <td>${num(row.ece, 3)}</td>
      </tr>`
    )
    .join("");

  $("weightTable").innerHTML = DATA.model.features
    .map(
      (feature) => `<tr>
        <td>${FEATURE_LABELS[feature][0]}</td>
        <td>${num(DATA.model.weights[feature], 4)}</td>
        <td>${FEATURE_LABELS[feature][1]}</td>
      </tr>`
    )
    .join("");

  $("calibrationChart").innerHTML = validation.calibration_bins
    .filter((bin) => bin.count)
    .map(
      (bin) => `<div class="calibration-row">
        <span>${bin.bin}</span>
        <div class="calibration-track">
          <span class="pred" style="width:${bin.avg_pred * 100}%"></span>
          <span class="actual" style="width:${bin.actual * 100}%"></span>
        </div>
        <span>${bin.count} 局</span>
      </div>`
    )
    .join("");
}

function renderDataPanel() {
  const dataset = DATA.dataset;
  $("sourceStatus").textContent = `${dataset.games.toLocaleString()} games · ${dataset.date_min.slice(0, 10)} to ${dataset.date_max.slice(0, 10)} · validation ${pct(DATA.model.validation.accuracy, 2)}`;
  $("sourceList").innerHTML = [
    `<div class="source-card"><strong>${DATA.sources.primary}</strong><span class="muted">${DATA.sources.download_endpoint}</span></div>`,
    ...DATA.sources.files.map(
      (file) => `<div class="source-card"><strong>${file.year} match data</strong><span class="muted">${file.file} · ${(file.bytes / 1024 / 1024).toFixed(1)} MB</span></div>`
    ),
    ...DATA.sources.notes.map((note) => `<div class="source-card"><strong>Note</strong><span class="muted">${note}</span></div>`),
  ].join("");
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("is-visible"));
      button.classList.add("is-active");
      $(button.dataset.panel).classList.add("is-visible");
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function init() {
  DATA = await fetch("platform_data.json").then((response) => response.json());
  hydrateTeamSelects();
  renderDraftControls();
  setDraftFromRosters();
  renderModelPanel();
  renderDataPanel();
  wireTabs();
  $("blueTeam").addEventListener("change", setDraftFromRosters);
  $("redTeam").addEventListener("change", setDraftFromRosters);
  $("seriesFormat").addEventListener("change", renderPrediction);
  $("resetDraft").addEventListener("click", setDraftFromRosters);
}

init().catch((error) => {
  $("sourceStatus").textContent = `Failed to load platform data: ${error.message}`;
});
