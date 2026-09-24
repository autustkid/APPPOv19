//  PPO v19 POLICY LIBRARY — 71-dim coast-aware PPO-trained autopush policy
//  Exposes: window.PPO19 = { ready, policy, loadError, version }
//  Weights loaded from ppo19-weights.json (same repo / CDN).

(function() {
    'use strict';

    const VERSION = '19.0.0';
    const WEIGHTS_URLS = [
        'https://cdn.jsdelivr.net/gh/autustkid/ppo19@main/ppo19-weights.json',
        'https://raw.githubusercontent.com/autustkid/ppo19/main/ppo19-weights.json',
        'https://raw.githack.com/autustkid/ppo19/main/ppo19-weights.json',
    ];

    // physics14 constants:
    const PS          = 35;
    const SPD         = 0.0016;
    const DELTA_NOM   = 1000 / 9;
    const DECAY_BASE  = 0.993;
    const TCR         = 45;
    const SPIKE_SCALE = 49;
    const KB          = 1.5;
    const SPIKE_DMG   = 15;

    // sparsecore15 constants:
    const STATE_DIM      = 71;
    const N_ACTIONS      = 73;
    const NULL_ACTION    = 72;
    const CONTACT_GOAL   = 8;
    const MAX_TICKS      = 25;
    const PUSH_LOOKAHEAD = 8;

    // physics14 (exact port):
    function computeSubsteps(xVel, yVel, delta) {
        const disp = Math.hypot(xVel * delta, yVel * delta);
        if (disp <= PS) return 1;
        return Math.ceil(disp / PS);
    }

    function simTick(state, moveDir, deltaOverride, baseSpdMOverride) {
        const DELTA = deltaOverride !== undefined && deltaOverride !== null ? deltaOverride : DELTA_NOM;
        const DF    = Math.pow(DECAY_BASE, DELTA);
        const bsM   = baseSpdMOverride !== undefined && baseSpdMOverride !== null ? baseSpdMOverride : 1.0;
        const s = {
            player: Object.assign({}, state.player),
            enemy:  Object.assign({}, state.enemy),
            trap:   state.trap,
            spikes: state.spikes,
            meFirst: state.meFirst,
        };
        const tookDamage = { player: false, enemy: false };
        const order      = s.meFirst ? [s.player, s.enemy] : [s.enemy, s.player];
        const isEnemyArr = s.meFirst ? [false, true]      : [true, false];
        for (let i = 0; i < 2; i++) {
            const p = order[i];
            const isEnemy = isEnemyArr[i];
            if (p.lockMove) { p.xVel = 0; p.yVel = 0; }
            if (!isEnemy && moveDir !== null && moveDir !== undefined && !Number.isNaN(moveDir)) {
                p.xVel += Math.cos(moveDir) * SPD * bsM * DELTA;
                p.yVel += Math.sin(moveDir) * SPD * bsM * DELTA;
            }
            p.lockMove = false;
            const nSub = computeSubsteps(p.xVel, p.yVel, DELTA);
            const seen = new Set();
            for (let sub = 0; sub < nSub; sub++) {
                p.x += p.xVel * DELTA / nSub;
                p.y += p.yVel * DELTA / nSub;
                const dT = Math.hypot(p.x - s.trap.x, p.y - s.trap.y);
                if (dT < TCR) p.lockMove = true;
                for (let si = 0; si < s.spikes.length; si++) {
                    if (seen.has(si)) continue;
                    const spike = s.spikes[si];
                    const Rs    = PS + spike.scale;
                    const dS    = Math.hypot(p.x - spike.x, p.y - spike.y);
                    if (dS < Rs && dS > 0.001) {
                        seen.add(si);
                        const ang = Math.atan2(p.y - spike.y, p.x - spike.x);
                        p.x = spike.x + Rs * Math.cos(ang);
                        p.y = spike.y + Rs * Math.sin(ang);
                        p.xVel *= 0.75;
                        p.yVel *= 0.75;
                        p.xVel += Math.cos(ang) * KB;
                        p.yVel += Math.sin(ang) * KB;
                        if (isEnemy) { p.hp = (p.hp || 100) - SPIKE_DMG; tookDamage.enemy = true; }
                        else { tookDamage.player = true; }
                    }
                }
            }
        }
        { const dx=s.enemy.x-s.player.x, dy=s.enemy.y-s.player.y, d=Math.hypot(dx,dy);
         if (d < 2*PS && d > 0.001) { const ov=2*PS-d, ux=dx/d, uy=dy/d;
                                     s.player.x -= (ov/2)*ux; s.player.y -= (ov/2)*uy;
                                     s.enemy.x  += (ov/2)*ux; s.enemy.y  += (ov/2)*uy; } }
        s.player.xVel *= DF; s.player.yVel *= DF;
        s.enemy.xVel  *= DF; s.enemy.yVel  *= DF;
        if (Math.abs(s.player.xVel) < 0.01) s.player.xVel = 0;
        if (Math.abs(s.player.yVel) < 0.01) s.player.yVel = 0;
        if (Math.abs(s.enemy.xVel)  < 0.01) s.enemy.xVel  = 0;
        if (Math.abs(s.enemy.yVel)  < 0.01) s.enemy.yVel  = 0;
        s.tookDamage = tookDamage;
        return s;
    }

    function simulateNoOp(state, nTicks) {
        let s = {
            player: { ...state.player },
            enemy:  { ...state.enemy },
            trap:   state.trap,
            spikes: state.spikes,
            meFirst: state.meFirst,
        };
        const traj = [{ x: s.enemy.x, y: s.enemy.y }];
        for (let i = 0; i < nTicks; i++) {
            const next = simTick(s, null, DELTA_NOM, state.baseSpdM ?? 1.0);
            s.player = next.player; s.enemy = next.enemy;
            traj.push({ x: s.enemy.x, y: s.enemy.y });
        }
        return { state: s, traj };
    }

    function simulateCoastPush(state, bestSpike, nTicks) {
        let s = {
            player: { ...state.player },
            enemy:  { ...state.enemy },
            trap:   state.trap,
            spikes: state.spikes,
            meFirst: state.meFirst,
        };
        const out = {
            traj: [{ x: s.enemy.x, y: s.enemy.y }],
            playerTraj: [{ x: s.player.x, y: s.player.y }],
            ttc: -1, contactPos: null, postBounceVel: null, contactSpeed: 0,
            endPlayer: null, midPlayer: null, endPlayerVel: 0,
        };
        const midIdx = Math.floor(nTicks / 2);
        for (let t = 0; t < nTicks; t++) {
            const preSpeed = Math.hypot(s.enemy.xVel || 0, s.enemy.yVel || 0);
            const nxt = simTick(s, null, DELTA_NOM, state.baseSpdM ?? 1.0);
            out.traj.push({ x: nxt.enemy.x, y: nxt.enemy.y });
            out.playerTraj.push({ x: nxt.player.x, y: nxt.player.y });
            if (out.ttc === -1 && nxt.tookDamage && nxt.tookDamage.enemy) {
                out.ttc = t + 1;
                out.contactPos = { x: nxt.enemy.x, y: nxt.enemy.y };
                out.postBounceVel = { x: nxt.enemy.xVel, y: nxt.enemy.yVel };
                out.contactSpeed = preSpeed;
            }
            s = { player: nxt.player, enemy: nxt.enemy, trap: nxt.trap, spikes: nxt.spikes, meFirst: nxt.meFirst };
            if (t + 1 === midIdx) out.midPlayer = { x: s.player.x, y: s.player.y };
        }
        out.endPlayer = { x: s.player.x, y: s.player.y };
        out.endPlayerVel = Math.hypot(s.player.xVel || 0, s.player.yVel || 0);
        return out;
    }

    function simulatePushAngle(state, bestSpike, nTicks, angle) {
        let s = {
            player: { ...state.player },
            enemy:  { ...state.enemy },
            trap:   state.trap,
            spikes: state.spikes,
            meFirst: state.meFirst,
        };
        const out = { ttc: -1, contactPos: null, postBounceVel: null, contactSpeed: 0 };
        for (let t = 0; t < nTicks; t++) {
            const preSpeed = Math.hypot(s.enemy.xVel || 0, s.enemy.yVel || 0);
            const nxt = simTick(s, angle, DELTA_NOM, state.baseSpdM ?? 1.0);
            if (out.ttc === -1 && nxt.tookDamage && nxt.tookDamage.enemy) {
                out.ttc = t + 1;
                out.contactPos = { x: nxt.enemy.x, y: nxt.enemy.y };
                out.postBounceVel = { x: nxt.enemy.xVel, y: nxt.enemy.yVel };
                out.contactSpeed = preSpeed;
                break;
            }
            s = { player: nxt.player, enemy: nxt.enemy, trap: nxt.trap, spikes: nxt.spikes, meFirst: nxt.meFirst };
        }
        return out;
    }

    function simulatePushStrategy(state, bestSpike, nTicks, strategy) {
        let s = {
            player: { ...state.player },
            enemy:  { ...state.enemy },
            trap:   state.trap,
            spikes: state.spikes,
            meFirst: state.meFirst,
        };
        const out = { traj: [{ x: s.enemy.x, y: s.enemy.y }], ttc: -1, contactPos: null, postBounceVel: null, contactSpeed: 0 };

        const dTS = Math.hypot(bestSpike.x - state.trap.x, bestSpike.y - state.trap.y);
        const axisX = dTS > 0.001 ? (bestSpike.x - state.trap.x) / dTS : 1;
        const axisY = dTS > 0.001 ? (bestSpike.y - state.trap.y) / dTS : 0;

        for (let t = 0; t < nTicks; t++) {
            let moveDir;
            if (strategy === 'direct') {
                const dx = s.enemy.x - s.player.x;
                const dy = s.enemy.y - s.player.y;
                moveDir = Math.hypot(dx, dy) > 0.001 ? Math.atan2(dy, dx) : 0;
            } else if (strategy === 'spikeAxis') {
                moveDir = Math.atan2(axisY, axisX);
            } else if (strategy === 'perpCW') {
                const dx = bestSpike.x - s.enemy.x;
                const dy = bestSpike.y - s.enemy.y;
                const m = Math.hypot(dx, dy);
                moveDir = m > 0.001 ? Math.atan2(-dx / m, dy / m) : 0;
            } else if (strategy === 'perpCCW') {
                const dx = bestSpike.x - s.enemy.x;
                const dy = bestSpike.y - s.enemy.y;
                const m = Math.hypot(dx, dy);
                moveDir = m > 0.001 ? Math.atan2(dx / m, -dy / m) : 0;
            } else {
                moveDir = 0;
            }

            const preSpeed = Math.hypot(s.enemy.xVel || 0, s.enemy.yVel || 0);
            const nxt = simTick(s, moveDir, DELTA_NOM, state.baseSpdM ?? 1.0);
            out.traj.push({ x: nxt.enemy.x, y: nxt.enemy.y });
            if (out.ttc === -1 && nxt.tookDamage && nxt.tookDamage.enemy) {
                out.ttc = t + 1;
                out.contactPos = { x: nxt.enemy.x, y: nxt.enemy.y };
                out.postBounceVel = { x: nxt.enemy.xVel, y: nxt.enemy.yVel };
                out.contactSpeed = preSpeed;
            }
            s = { player: nxt.player, enemy: nxt.enemy, trap: nxt.trap, spikes: nxt.spikes, meFirst: nxt.meFirst };
        }
        return out;
    }

    function playerStoppingModel(vx, vy) {
        const v0 = Math.hypot(vx, vy);
        if (v0 < 0.01) return { ticks: 0, distance: 0 };
        const perTickDecay = Math.pow(DECAY_BASE, DELTA_NOM);
        const ticks = Math.log(0.01 / v0) / Math.log(perTickDecay);
        const distance = v0 * DELTA_NOM * (1 - Math.pow(perTickDecay, Math.max(0, ticks))) / (1 - perTickDecay);
        return { ticks: Math.max(0, ticks), distance: Math.max(0, distance) };
    }

    function bounceEscapeDanger(state, bestSpike) {
        const { player, enemy, trap } = state;
        const enToTrap = Math.hypot(enemy.x - trap.x, enemy.y - trap.y);
        if (enToTrap < TCR * 0.85) return 0;

        const peX = enemy.x - player.x, peY = enemy.y - player.y;
        const peMag = Math.hypot(peX, peY);
        if (peMag < 0.001) return 0;
        const esX = bestSpike.x - enemy.x, esY = bestSpike.y - enemy.y;
        if (peX * esX + peY * esY <= 0) return 0;

        const teX = enemy.x - trap.x, teY = enemy.y - trap.y;
        const pushDirX = peX / peMag, pushDirY = peY / peMag;
        if (teX * pushDirX + teY * pushDirY <= 0) return 0;

        return 1;
    }

    // sparsecore15 encoder (exact port):
    function encodeState(state, prevAction = null, ticksElapsed = 0) {
        const { player, enemy, trap, spikes, meFirst, baseSpdM } = state;

        let bestSpike = null, bestDepth = Infinity;
        for (const s of spikes) {
            const d = Math.hypot(s.x - trap.x, s.y - trap.y);
            if (d < 0.001 || d > TCR + PS + s.scale) continue;
            const Rs = PS + s.scale;
            const P_dx = s.x - Rs * (s.x - trap.x) / d - trap.x;
            const P_dy = s.y - Rs * (s.y - trap.y) / d - trap.y;
            const depth = Math.hypot(P_dx, P_dy);
            if (depth < bestDepth) { bestDepth = depth; bestSpike = s; }
        }
        if (!bestSpike) return null;

        const thetaSpike = Math.atan2(bestSpike.y - trap.y, bestSpike.x - trap.x);
        const cosR = Math.cos(-thetaSpike), sinR = Math.sin(-thetaSpike);
        const rot = (x, y) => ({ x: x * cosR - y * sinR, y: x * sinR + y * cosR });

        const spikeRel = rot(bestSpike.x - trap.x, bestSpike.y - trap.y);
        const enRel    = rot(enemy.x - trap.x, enemy.y - trap.y);
        const enVel    = rot(enemy.xVel || 0, enemy.yVel || 0);
        const plRel    = rot(player.x - trap.x, player.y - trap.y);
        const plVel    = rot(player.xVel || 0, player.yVel || 0);

        const enToTrap = Math.hypot(enemy.x - trap.x, enemy.y - trap.y);
        const fDistTrap = Math.max(0, (TCR - enToTrap) / TCR);
        const enToSpike = Math.hypot(enemy.x - bestSpike.x, enemy.y - bestSpike.y);
        const Rs_best = PS + bestSpike.scale;
        let fDistSpike = (Rs_best - enToSpike) / Rs_best;
        if (fDistSpike > 1) fDistSpike = 1;
        if (fDistSpike < -1) fDistSpike = -1;

        const esX = enemy.x - bestSpike.x, esY = enemy.y - bestSpike.y;
        const esMag = Math.hypot(esX, esY);
        let YsX, YsY;
        if (esMag > 0.001) {
            const ux = esX / esMag, uy = esY / esMag;
            YsX = enemy.x + ux * 66;
            YsY = enemy.y + uy * 66;
        } else { YsX = player.x; YsY = player.y; }
        const plToY = Math.hypot(player.x - YsX, player.y - YsY);
        const fPlToY = plToY / 100;

        const scr = PS + bestSpike.scale;
        const dSpT = Math.hypot(bestSpike.x - trap.x, bestSpike.y - trap.y);
        let optX, optY;
        if (dSpT > 0.001) {
            const tdirX = (trap.x - bestSpike.x) / dSpT, tdirY = (trap.y - bestSpike.y) / dSpT;
            optX = bestSpike.x + tdirX * scr;
            optY = bestSpike.y + tdirY * scr;
        } else { optX = bestSpike.x; optY = bestSpike.y; }

        const vEnToOpt = rot(optX - enemy.x, optY - enemy.y);

        const pushMag = Math.hypot(optX - enemy.x, optY - enemy.y);
        let pushX, pushY;
        if (pushMag > 0.001) { pushX = (optX - enemy.x) / pushMag; pushY = (optY - enemy.y) / pushMag; }
        else { pushX = 1; pushY = 0; }

        const ramBackDist = 2 * PS * 0.85;
        const idealX = enemy.x - pushX * ramBackDist;
        const idealY = enemy.y - pushY * ramBackDist;
        const vPlToIdeal = rot(idealX - player.x, idealY - player.y);
        const distFromIdeal = Math.hypot(idealX - player.x, idealY - player.y);

        const drift1 = simulateNoOp(state, 1);
        const drift5 = simulateNoOp(state, 5);
        const vDrift1 = rot(drift1.state.enemy.x - trap.x, drift1.state.enemy.y - trap.y);
        const vDrift5 = rot(drift5.state.enemy.x - trap.x, drift5.state.enemy.y - trap.y);
        const drift5ToTrap = Math.hypot(drift5.state.enemy.x - trap.x, drift5.state.enemy.y - trap.y);
        const driftSafeIn5 = drift5ToTrap < TCR ? 1 : 0;

        const pushSim = simulatePushStrategy(state, bestSpike, PUSH_LOOKAHEAD, 'direct');
        const pushTTCNorm = pushSim.ttc === -1 ? 1.0 : pushSim.ttc / PUSH_LOOKAHEAD;
        const t3Idx = Math.min(3, pushSim.traj.length - 1);
        const tEndIdx = pushSim.traj.length - 1;
        const vPushT3 = rot(pushSim.traj[t3Idx].x - trap.x, pushSim.traj[t3Idx].y - trap.y);
        const vPushEnd = rot(pushSim.traj[tEndIdx].x - trap.x, pushSim.traj[tEndIdx].y - trap.y);
        let vPushContact = vPushEnd;
        let bounceAlign = 0;
        if (pushSim.contactPos) {
            vPushContact = rot(pushSim.contactPos.x - trap.x, pushSim.contactPos.y - trap.y);
            const vel = pushSim.postBounceVel;
            const vMag = Math.hypot(vel.x, vel.y);
            if (vMag > 0.001) {
                const spTrX = trap.x - bestSpike.x;
                const spTrY = trap.y - bestSpike.y;
                const spTrMag = Math.hypot(spTrX, spTrY);
                if (spTrMag > 0.001) {
                    bounceAlign = (vel.x * spTrX + vel.y * spTrY) / (vMag * spTrMag);
                    if (bounceAlign > 1) bounceAlign = 1;
                    if (bounceAlign < -1) bounceAlign = -1;
                }
            }
        }

        const coastSim = simulateCoastPush(state, bestSpike, PUSH_LOOKAHEAD);
        const coastTTCNorm = coastSim.ttc === -1 ? 1.0 : coastSim.ttc / PUSH_LOOKAHEAD;
        let coastContactQuality = 0, coastUseful = 0;
        if (coastSim.contactPos) {
            coastContactQuality = Math.max(0, 1 - coastSim.contactSpeed / 0.5);
            const dCT = Math.hypot(coastSim.contactPos.x - trap.x, coastSim.contactPos.y - trap.y);
            coastUseful = (dCT <= TCR) ? 1 : 0;
        }

        const coastEndRelRaw = coastSim.endPlayer
            ? rot(coastSim.endPlayer.x - trap.x, coastSim.endPlayer.y - trap.y)
            : rot(player.x - trap.x, player.y - trap.y);
        const coastDistToIdealEnd = coastSim.endPlayer
            ? Math.hypot(coastSim.endPlayer.x - idealX, coastSim.endPlayer.y - idealY)
            : distFromIdeal;
        const plSpeedNow = Math.hypot(player.xVel || 0, player.yVel || 0);
        const coastEndSpeedFrac = plSpeedNow > 0.001
            ? Math.min(1, coastSim.endPlayerVel / plSpeedNow)
            : 0;

        let nextNullPosRel = rot(player.x - trap.x, player.y - trap.y);
        let nextNullSpeedFrac = 0;
        let nextNullDistToEnemy = Math.hypot(enemy.x - player.x, enemy.y - player.y) / 100;
        {
            const oneTick = simTick(state, null, DELTA_NOM, baseSpdM ?? 1.0);
            nextNullPosRel = rot(oneTick.player.x - trap.x, oneTick.player.y - trap.y);
            const nextSpeed = Math.hypot(oneTick.player.xVel || 0, oneTick.player.yVel || 0);
            nextNullSpeedFrac = plSpeedNow > 0.001
                ? Math.min(1, nextSpeed / plSpeedNow)
                : 0;
            nextNullDistToEnemy = Math.min(2, Math.hypot(oneTick.enemy.x - oneTick.player.x, oneTick.enemy.y - oneTick.player.y) / 100);
        }

        let coastWillOvershoot = 0;
        if (coastSim.endPlayer) {
            coastWillOvershoot = (coastDistToIdealEnd > distFromIdeal + 1) ? 1 : 0;
        }

        let contactWindowAngleNorm = 0;
        {
            const sDist = Math.hypot(bestSpike.x - trap.x, bestSpike.y - trap.y);
            const Rs = PS + (bestSpike.scale ?? SPIKE_SCALE);
            if (sDist > 0.001 && Rs > 0.001) {
                const cosArg = (sDist*sDist + Rs*Rs - TCR*TCR) / (2 * sDist * Rs);
                if (cosArg > -1 && cosArg < 1) {
                    const fullArc = 2 * Math.acos(cosArg);
                    contactWindowAngleNorm = Math.min(1, fullArc / Math.PI);
                } else if (cosArg >= 1) {
                    contactWindowAngleNorm = 0;
                } else {
                    contactWindowAngleNorm = 1;
                }
            }
        }

        let minPushOffsetNorm = 0;
        {
            const esX = bestSpike.x - enemy.x, esY = bestSpike.y - enemy.y;
            const esMag = Math.hypot(esX, esY);
            if (esMag > 0.001) {
                const uX = esX / esMag, uY = esY / esMag;
                const pX = player.x - enemy.x, pY = player.y - enemy.y;
                const dot = pX * uX + pY * uY;
                const perpX = pX - dot * uX, perpY = pY - dot * uY;
                minPushOffsetNorm = Math.min(2, Math.hypot(perpX, perpY) / 50);
            }
        }

        let ticksToIdealMomentum = 1.0;
        {
            if (distFromIdeal > 0.5) {
                const toIdX = idealX - player.x, toIdY = idealY - player.y;
                const toIdMag = Math.hypot(toIdX, toIdY);
                if (toIdMag > 0.001) {
                    const uIX = toIdX / toIdMag, uIY = toIdY / toIdMag;
                    const momentumTowardIdeal = (player.xVel || 0) * uIX + (player.yVel || 0) * uIY;
                    const nominalPush = 0.15 * (baseSpdM ?? 1.0);
                    const effectiveSpeed = Math.max(0.02, nominalPush + Math.max(0, momentumTowardIdeal));
                    ticksToIdealMomentum = Math.min(1, (distFromIdeal / effectiveSpeed) / 10);
                }
            } else {
                ticksToIdealMomentum = 0;
            }
        }

        let enemyDriftOutImminent = 0;
        {
            const drift3Traj = drift5.traj;
            for (let i = 1; i <= 3 && i < drift3Traj.length; i++) {
                const dx = drift3Traj[i].x - trap.x, dy = drift3Traj[i].y - trap.y;
                if (Math.hypot(dx, dy) > TCR) { enemyDriftOutImminent = 1; break; }
            }
        }

        let maxPushQualityAny = 0;
        {
            const nSweep = 8;
            for (let k = 0; k < nSweep; k++) {
                const ang = (k / nSweep) * 2 * Math.PI;
                const sim = simulatePushAngle(state, bestSpike, PUSH_LOOKAHEAD, ang);
                if (sim.contactPos) {
                    const dCT = Math.hypot(sim.contactPos.x - trap.x, sim.contactPos.y - trap.y);
                    if (dCT <= TCR) {
                        const q = Math.max(0, 1 - sim.contactSpeed / 0.5);
                        if (q > maxPushQualityAny) maxPushQualityAny = q;
                    }
                }
            }
        }

        let coastNearestApproachNorm = 1;
        {
            const Rs = PS + (bestSpike.scale ?? SPIKE_SCALE);
            let minD = Infinity;
            for (const p of coastSim.playerTraj) {
                const d = Math.hypot(p.x - bestSpike.x, p.y - bestSpike.y) - Rs;
                if (d < minD) minD = d;
            }
            coastNearestApproachNorm = Math.min(2, Math.max(0, minD / 50));
        }

        function alternateFeatures(strat) {
            const sim = simulatePushStrategy(state, bestSpike, PUSH_LOOKAHEAD, strat);
            const ttcN = sim.ttc === -1 ? 1.0 : sim.ttc / PUSH_LOOKAHEAD;
            let bAlign = 0;
            if (sim.contactPos && sim.postBounceVel) {
                const vel = sim.postBounceVel;
                const vMag = Math.hypot(vel.x, vel.y);
                if (vMag > 0.001) {
                    const spTrX = trap.x - bestSpike.x;
                    const spTrY = trap.y - bestSpike.y;
                    const spTrMag = Math.hypot(spTrX, spTrY);
                    if (spTrMag > 0.001) {
                        bAlign = (vel.x * spTrX + vel.y * spTrY) / (vMag * spTrMag);
                        if (bAlign > 1) bAlign = 1;
                        if (bAlign < -1) bAlign = -1;
                    }
                }
            }
            const end = sim.traj[sim.traj.length - 1];
            const endDist = Math.hypot(end.x - trap.x, end.y - trap.y) / TCR;
            return [ttcN, bAlign, Math.min(2, endDist)];
        }
        const altAxis = alternateFeatures('spikeAxis');
        const altCW = alternateFeatures('perpCW');
        const altCCW = alternateFeatures('perpCCW');

        const bounceDanger = bounceEscapeDanger(state, bestSpike);

        const enVelMag = Math.hypot(enemy.xVel || 0, enemy.yVel || 0);
        const enToOptMag = Math.hypot(optX - enemy.x, optY - enemy.y);
        let enVelAlignOpt = 0;
        if (enVelMag > 0.01 && enToOptMag > 0.1) {
            const uoX = (optX - enemy.x) / enToOptMag;
            const uoY = (optY - enemy.y) / enToOptMag;
            const uvX = (enemy.xVel || 0) / enVelMag;
            const uvY = (enemy.yVel || 0) / enVelMag;
            enVelAlignOpt = uoX * uvX + uoY * uvY;
        }

        const stop = playerStoppingModel(player.xVel || 0, player.yVel || 0);
        const plStopTicksNorm = Math.min(1, stop.ticks / 10);
        const plStopDistNorm = Math.min(1, stop.distance / 100);

        let minDriftDist = Infinity, maxDriftDist = 0, ticksUntilEscape = 5;
        for (let i = 0; i < drift5.traj.length; i++) {
            const d = Math.hypot(drift5.traj[i].x - trap.x, drift5.traj[i].y - trap.y);
            if (d < minDriftDist) minDriftDist = d;
            if (d > maxDriftDist) maxDriftDist = d;
            if (d > TCR + 0.5 && ticksUntilEscape === 5) ticksUntilEscape = i;
        }
        const minDriftNorm = minDriftDist / TCR;
        const maxDriftNorm = Math.min(2, maxDriftDist / TCR);
        const ticksUntilEscapeNorm = ticksUntilEscape / 5;

        const ticksRemainingNorm = Math.max(0, (MAX_TICKS - ticksElapsed) / MAX_TICKS);

        let contactQuality = 0;
        if (pushSim.contactPos) {
            contactQuality = Math.max(0, 1 - pushSim.contactSpeed / 0.5);
        }

        const plVelMag = Math.hypot(player.xVel || 0, player.yVel || 0);
        let plMomentumAlign = 0;
        if (plVelMag > 0.01) {
            const uvX = (player.xVel || 0) / plVelMag;
            const uvY = (player.yVel || 0) / plVelMag;
            plMomentumAlign = uvX * pushX + uvY * pushY;
        }

        let prevCos = 0, prevSin = 0;
        if (prevAction !== null && prevAction !== undefined && prevAction !== NULL_ACTION) {
            const worldAngle = (prevAction / 72) * 2 * Math.PI;
            const localAngle = worldAngle - thetaSpike;
            prevCos = Math.cos(localAngle);
            prevSin = Math.sin(localAngle);
        }

        return {
            features: [
                spikeRel.x / 100,
                enRel.x / 50,  enRel.y / 50,
                enVel.x / 0.5, enVel.y / 0.5,
                plRel.x / 100, plRel.y / 100,
                plVel.x / 1,   plVel.y / 1,
                meFirst ? 1 : -1,
                fDistTrap,
                fDistSpike,
                fPlToY,
                (baseSpdM ?? 1.0),
                vEnToOpt.x / 50, vEnToOpt.y / 50,
                vPlToIdeal.x / 100, vPlToIdeal.y / 100,
                distFromIdeal / 100,
                vDrift1.x / 50, vDrift1.y / 50,
                vDrift5.x / 50, vDrift5.y / 50,
                driftSafeIn5,
                pushTTCNorm,
                vPushT3.x / 50, vPushT3.y / 50,
                vPushEnd.x / 50, vPushEnd.y / 50,
                vPushContact.x / 50, vPushContact.y / 50,
                bounceAlign,
                altAxis[0], altAxis[1], altAxis[2],
                altCW[0],   altCW[1],   altCW[2],
                altCCW[0],  altCCW[1],  altCCW[2],
                bounceDanger,
                enVelAlignOpt,
                plStopTicksNorm, plStopDistNorm,
                minDriftNorm, maxDriftNorm, ticksUntilEscapeNorm,
                ticksRemainingNorm,
                contactQuality,
                plMomentumAlign,
                prevCos, prevSin,
                coastTTCNorm, coastContactQuality, coastUseful,
                coastEndRelRaw.x / 100, coastEndRelRaw.y / 100,
                Math.min(2, coastDistToIdealEnd / 100),
                coastEndSpeedFrac,
                coastWillOvershoot,
                contactWindowAngleNorm,
                minPushOffsetNorm,
                ticksToIdealMomentum,
                enemyDriftOutImminent,
                maxPushQualityAny,
                coastNearestApproachNorm,
                nextNullPosRel.x / 100, nextNullPosRel.y / 100,
                nextNullSpeedFrac,
                nextNullDistToEnemy,
            ],
            rotation: thetaSpike,
            bestSpike,
            optX, optY,
        };
    }

    function decodeAction(actionIdx, rotation) {
        if (actionIdx === NULL_ACTION) return null;
        return (actionIdx / 72) * 2 * Math.PI + rotation;
    }

    // PolicyNet:
    class PolicyNet {
        constructor(h1 = 96, h2 = 96) { this.h1 = h1; this.h2 = h2; }
        load(d) {
            this.h1 = d.h1 || 96;
            this.h2 = d.h2 || 96;
            this.W1 = Float64Array.from(d.W1); this.b1 = Float64Array.from(d.b1);
            this.W2 = Float64Array.from(d.W2); this.b2 = Float64Array.from(d.b2);
            this.W3 = Float64Array.from(d.W3); this.b3 = Float64Array.from(d.b3);
        }
        forward(x) {
            const h1 = new Float64Array(this.h1);
            for (let j = 0; j < this.h1; j++) {
                let s = this.b1[j];
                for (let i = 0; i < STATE_DIM; i++) s += x[i] * this.W1[i * this.h1 + j];
                h1[j] = Math.tanh(s);
            }
            const h2 = new Float64Array(this.h2);
            for (let j = 0; j < this.h2; j++) {
                let s = this.b2[j];
                for (let i = 0; i < this.h1; i++) s += h1[i] * this.W2[i * this.h2 + j];
                h2[j] = Math.tanh(s);
            }
            const logits = new Float64Array(N_ACTIONS);
            for (let j = 0; j < N_ACTIONS; j++) {
                let s = this.b3[j];
                for (let i = 0; i < this.h2; i++) s += h2[i] * this.W3[i * N_ACTIONS + j];
                logits[j] = s;
            }
            let maxL = -Infinity;
            for (let j = 0; j < N_ACTIONS; j++) if (logits[j] > maxL) maxL = logits[j];
            const probs = new Float64Array(N_ACTIONS);
            let sum = 0;
            for (let j = 0; j < N_ACTIONS; j++) { probs[j] = Math.exp(logits[j] - maxL); sum += probs[j]; }
            for (let j = 0; j < N_ACTIONS; j++) probs[j] /= sum;
            return { logits, probs };
        }
    }

    // Weights loader:
    let _policy = null;
    let _loadError = null;

    async function loadPolicy() {
        for (let i = 0; i < WEIGHTS_URLS.length; i++) {
            const url = WEIGHTS_URLS[i];
            try {
                console.log('[PPO19] loading weights:', url);
                const t0 = performance.now();
                const r = await fetch(url, { cache: 'force-cache' });
                if (!r.ok) throw new Error('HTTP ' + r.status);
                const w = await r.json();
                const net = new PolicyNet();
                net.load(w);
                _policy = net;
                const ms = (performance.now() - t0).toFixed(0);
                console.log(`[PPO19] ready in ${ms}ms: h1=${net.h1} h2=${net.h2} STATE_DIM=${STATE_DIM}`);
                PPO19.ready = true;
                return;
            } catch (err) {
                console.warn('[PPO19] load failed from', url, '→', err.message);
                _loadError = err;
            }
        }
        console.error('[PPO19] ALL weight sources failed. Policy disabled.');
        PPO19.loadError = _loadError;
    }

    // Persistent state:
    let _prevAction   = null;
    let _tickCount    = 0;
    let _lastTrapKey  = null;
    let _lastEnemyKey = null;

    // Public API:
    const PPO19 = {
        version: VERSION,
        ready: false,
        loadError: null,

        /**
         * @param {object} player  { x, y, xVel?, yVel? }
         * @param {object} enemy   { x, y, xVel?, yVel?, sid? }
         * @param {object} trap    { x, y, sid? }
         * @param {Array}  spikes  [{ x, y, scale? }]
         * @param {boolean} meFirst
         * @param {number} baseSpdM
         * @returns {{ moveDir: number|null, action: number, loading?: boolean }}
         */
        policy(player, enemy, trap, spikes, meFirst, baseSpdM) {
            if (!PPO19.ready || !_policy) {
                return { moveDir: null, action: NULL_ACTION, loading: true };
            }

            const trapKey  = (trap && trap.sid != null) ? trap.sid : ((trap.x|0)+'|'+(trap.y|0));
            const enemyKey = (enemy && enemy.sid != null) ? enemy.sid : null;
            if (trapKey !== _lastTrapKey || enemyKey !== _lastEnemyKey) {
                _prevAction = null; _tickCount = 0;
                _lastTrapKey = trapKey; _lastEnemyKey = enemyKey;
            }

            const state = {
                player:  { x: player.x, y: player.y, xVel: player.xVel || 0, yVel: player.yVel || 0, lockMove: false },
                enemy:   { x: enemy.x,  y: enemy.y,  xVel: enemy.xVel  || 0, yVel: enemy.yVel  || 0, lockMove: false },
                trap:    { x: trap.x,   y: trap.y },
                spikes:  (spikes || []).map(function (s) { return { x: s.x, y: s.y, scale: s.scale || SPIKE_SCALE }; }),
                meFirst: !!meFirst,
                baseSpdM: baseSpdM != null ? baseSpdM : 1.0,
            };

            const enc = encodeState(state, _prevAction, _tickCount);
            if (!enc || !enc.features || enc.features.length !== STATE_DIM) {
                return { moveDir: null, action: NULL_ACTION };
            }

            const fwd = _policy.forward(enc.features);
            let action = 0;
            for (let j = 1; j < N_ACTIONS; j++) if (fwd.probs[j] > fwd.probs[action]) action = j;

            const moveDir = decodeAction(action, enc.rotation);
            _prevAction = action;
            _tickCount  = (_tickCount + 1) % MAX_TICKS;

            return { moveDir: moveDir, action: action };
        },

        _reload() {
            PPO19.ready = false; return loadPolicy();
        },
    };

    window.PPO19 = PPO19;
    console.log('[PPO19] library v' + VERSION + ' loaded, fetching weights...');
    loadPolicy();
})();
