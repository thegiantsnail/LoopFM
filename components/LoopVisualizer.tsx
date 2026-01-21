import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// --- Configuration Constants ---
const NUM_CTRL = 10;
const SAMPLE_N = 900;
const MOD_SAMPLE_N = 900;
const RADIUS = 1.0;

// --- Helper Types ---
type ColorMode = 'phase' | 'curvature';
type ViewMode = '2d' | '3d';

interface VisualizerState {
  colorMode: ColorMode;
  viewMode: ViewMode;
  zScale: number;
  speed: number;
  tension: number;
  isPlaying: boolean;
  showPoints: boolean;
  showArrows: boolean;
  // Dual / Math features
  showDual: boolean;
  showSum: boolean;
  showChord: boolean;
  dualOffset: number;
  unisonWidth: number; // Renamed from chordPitch
  stereoPhase: number;
  // Audio features
  audioEnabled: boolean;
  audioFreq: number;
  audioGain: number;
  frenetFM: number;
  frenetStereo: number;
  // Topological FM
  topoFM: number;
  topoSpeed: number;
  showModulator: boolean;
}

interface Stats {
  winding: number;
}

// --- Math Helpers ---
function hsvToRgb(h: number, s: number, v: number) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r = 0, g = 0, b = 0;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return { r, g, b };
}

function rampCurvatureSigned(x: number) {
  const t = Math.max(-1, Math.min(1, x));
  let r, g, b;

  if (t < 0) {
    const val = 1 + t;
    r = 0.1 + 0.9 * val;
    g = 0.4 + 0.6 * val;
    b = 1.0;
  } else {
    const val = t;
    r = 1.0;
    g = 1.0 - 0.9 * val;
    b = 1.0 - 0.8 * val;
  }
  return { r, g, b };
}

const LoopVisualizer: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);

  // -- Refs for Visualization Overlay --
  const scopeCanvasRef = useRef<HTMLCanvasElement>(null);
  const distortCanvasRef = useRef<HTMLCanvasElement>(null);
  const harmonicsRef = useRef<HTMLDivElement>(null);
  const varianceRef = useRef<HTMLDivElement>(null);

  // -- React State for UI --
  const [config, setConfig] = useState<VisualizerState>({
    colorMode: 'phase',
    viewMode: '2d',
    zScale: 1.15,
    speed: 1.0,
    tension: 0.45,
    isPlaying: true,
    showPoints: true,
    showArrows: true,
    showDual: false,
    showSum: false,
    showChord: false,
    dualOffset: 0.0,
    unisonWidth: 0.0,
    stereoPhase: 0.0,
    audioEnabled: false,
    audioFreq: 110,
    audioGain: 0.15,
    frenetFM: 0.0,
    frenetStereo: 0.5,
    topoFM: 0.0,
    topoSpeed: 1.0,
    showModulator: false,
  });

  const [stats, setStats] = useState<Stats>({ winding: 1.0 });

  // -- Audio Refs --
  const audioCtxRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const audioPhaseRef = useRef<number>(0);
  const audioPhaseRefR = useRef<number>(0); // Independent phase for Right/Unison channel
  const modPhaseRef = useRef<number>(0);

  // -- Scene Context --
  const sceneContext = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    curve: THREE.CatmullRomCurve3 | null;
    ctrlPoints: THREE.Mesh[];
    ctrlBase: THREE.Vector3[];
    ctrlOffsets: THREE.Vector3[];
    lineGeom: THREE.BufferGeometry;
    modLine: THREE.Line; // Modulator Loop Visual
    particle: THREE.Mesh;
    particle2: THREE.Mesh; 
    sumParticle: THREE.Mesh;
    chordLine: THREE.Line;
    grid: THREE.GridHelper;
    ctrlGroup: THREE.Group;
    arrowGroup: THREE.Group;
    sampled: THREE.Vector3[];
    modSampled: THREE.Vector3[]; // Modulator Geometry Data
    tangents: THREE.Vector3[];
    normals: THREE.Vector3[];
    binormals: THREE.Vector3[];
    phaseNorm: Float32Array;
    curvatureNorm: Float32Array;
    dragging: boolean;
    dragIndex: number;
    time: number;
    config: VisualizerState;
    setStats: (s: Stats) => void;
    // Audio-Visual Sync
    visualChordDistance: number;
    smoothedDist: number;
    // Real-time Audio Data for Vectorscope (increased buffer size)
    audioVizData: { left: Float32Array; right: Float32Array };
  } | null>(null);

  // Sync React state to the mutable config ref & Update Audio
  useEffect(() => {
    if (sceneContext.current) {
      const ctx = sceneContext.current;
      const prevConfig = ctx.config;
      ctx.config = config;
      ctx.setStats = setStats;

      // Update Gain Realtime
      if (gainNodeRef.current && audioCtxRef.current) {
        gainNodeRef.current.gain.setTargetAtTime(
            config.audioEnabled ? config.audioGain : 0, 
            audioCtxRef.current.currentTime, 
            0.1
        );
      }

      let needsCurveRebuild = false;
      let needsAttributeUpdate = false;
      let needsViewSync = false;

      if (prevConfig.tension !== config.tension) needsCurveRebuild = true;
      if (prevConfig.showPoints !== config.showPoints) ctx.ctrlGroup.visible = config.showPoints;
      if (prevConfig.showArrows !== config.showArrows) {
        ctx.arrowGroup.visible = config.showArrows;
        if (config.showArrows) needsAttributeUpdate = true;
      }
      
      ctx.modLine.visible = config.showModulator;
      ctx.particle2.visible = config.showDual;
      ctx.sumParticle.visible = config.showDual && config.showSum;
      ctx.chordLine.visible = config.showDual && config.showChord;

      if (
        prevConfig.colorMode !== config.colorMode ||
        prevConfig.viewMode !== config.viewMode ||
        prevConfig.zScale !== config.zScale
      ) {
        needsAttributeUpdate = true;
      }
      
      if (prevConfig.viewMode !== config.viewMode) {
        needsViewSync = true;
      }

      if (needsCurveRebuild) rebuildCurve(ctx);
      if (needsCurveRebuild || needsAttributeUpdate) updateSamplesAndAttributes(ctx);
      if (needsViewSync) syncViewMode(ctx);
    }
  }, [config]);


  // --- Audio Logic ---
  const toggleAudio = () => {
      if (!config.audioEnabled) {
          // Init Audio
          const Ctx = window.AudioContext || (window as any).webkitAudioContext;
          const ctx = new Ctx();
          audioCtxRef.current = ctx;

          const gain = ctx.createGain();
          gain.gain.value = config.audioGain;
          gain.connect(ctx.destination);
          gainNodeRef.current = gain;

          const bufferSize = 4096;
          const processor = ctx.createScriptProcessor(bufferSize, 0, 2);
          
          processor.onaudioprocess = (e) => {
              const L = e.outputBuffer.getChannelData(0);
              const R = e.outputBuffer.getChannelData(1);
              const sCtx = sceneContext.current;
              
              if (!sCtx) {
                  L.fill(0);
                  R.fill(0);
                  return;
              }

              const baseFreq = sCtx.config.audioFreq;
              const sampleRate = ctx.sampleRate;
              
              const beta = sCtx.config.frenetFM;
              const stereoAmt = sCtx.config.frenetStereo;
              const topoFM = sCtx.config.topoFM;
              const topoSpeed = sCtx.config.topoSpeed;
              const unisonWidth = sCtx.config.unisonWidth;
              const stereoPhase = sCtx.config.stereoPhase;
              
              const N = SAMPLE_N;
              const M = MOD_SAMPLE_N;
              const tangents = sCtx.tangents;
              const normals = sCtx.normals;
              const binormals = sCtx.binormals;
              const modSamples = sCtx.modSampled;

              const dt = 1.0 / sampleRate;

              for (let i = 0; i < bufferSize; i++) {
                  // --- Control Smoothing ---
                  const targetDist = sCtx.visualChordDistance;
                  sCtx.smoothedDist += (targetDist - sCtx.smoothedDist) * 0.005; 
                  const dist = sCtx.smoothedDist;

                  // --- 1. Topological FM Update (Modulator LFO) ---
                  modPhaseRef.current += (baseFreq * topoSpeed * dt);
                  if (modPhaseRef.current >= 1.0) modPhaseRef.current -= 1.0;

                  let modVal = 0;
                  if (topoFM > 0.001) {
                      const mIdx = Math.floor(modPhaseRef.current * M) % M;
                      modVal = modSamples[mIdx].x; 
                  }

                  // --- 2. Advance Carrier Phases (Unison / Detune) ---
                  // Calculate Unison Detune Factor based on visual chord distance
                  let detuneMult = 0;
                  if (unisonWidth > 0.001) {
                      detuneMult = dist * unisonWidth * 0.03; // Scale factor for detune intensity
                  }

                  const freqL = baseFreq; 
                  const freqR = baseFreq * (1.0 + detuneMult); // R channel drifts when chord expands

                  audioPhaseRef.current += (freqL * dt);
                  audioPhaseRefR.current += (freqR * dt);

                  if (audioPhaseRef.current >= 1.0) audioPhaseRef.current -= 1.0;
                  if (audioPhaseRefR.current >= 1.0) audioPhaseRefR.current -= 1.0;

                  // Apply Topo Phase Mod (modulates both read pointers)
                  let pL = audioPhaseRef.current;
                  let pR = audioPhaseRefR.current;

                  if (topoFM > 0.001) {
                     const topoShift = (topoFM * modVal * 0.2);
                     pL += topoShift;
                     pR += topoShift;
                  }
                  
                  // Wrap phases after mod
                  pL -= Math.floor(pL); if (pL < 0) pL += 1.0;
                  pR -= Math.floor(pR); if (pR < 0) pR += 1.0;

                  // --- 3. Stereo Phase Offset (Static) ---
                  // Additional static phase offset on R based on distance
                  if (stereoPhase > 0.001) {
                      pR += dist * stereoPhase * 0.05;
                      pR -= Math.floor(pR); if (pR < 0) pR += 1.0;
                  }

                  // --- 4. Synthesis: Frenet Phase Modulation ---
                  // Base Index for L lookup
                  const idxL_base = Math.floor(pL * N);
                  // Interpolate Normal/Binormal for modulation
                  const n1 = normals[idxL_base % N];
                  const b1 = binormals[idxL_base % N];
                  // Ideally we interpolate, but nearest neighbor for modulation source is okay for performance here
                  const Nx_mod = n1.x; 
                  const Ny_mod = n1.y; // Use Y component of normal for R mod
                  const Bz_mod = b1.z;

                  // Modulate Phase by Normal component (Frenet FM)
                  let readPhL = pL + (Nx_mod * beta * 0.2); 
                  let readPhR = pR + (Ny_mod * beta * 0.2); 

                  // Wrap read phases
                  readPhL -= Math.floor(readPhL); if(readPhL < 0) readPhL += 1;
                  readPhR -= Math.floor(readPhR); if(readPhR < 0) readPhR += 1;

                  // Sample Tangents (Carrier Waveform)
                  const idxL = Math.floor(readPhL * N);
                  const nextIdxL = (idxL + 1) % N;
                  const tL = (readPhL * N) - idxL;
                  const valL = tangents[idxL].x * (1-tL) + tangents[nextIdxL].x * tL;

                  const idxR = Math.floor(readPhR * N);
                  const nextIdxR = (idxR + 1) % N;
                  const tR = (readPhR * N) - idxR;
                  const valR = tangents[idxR].y * (1-tR) + tangents[nextIdxR].y * tR;

                  // --- 5. Spatialization: Binormal Stereo Rotation ---
                  const rotAngle = Bz_mod * stereoAmt * Math.PI * 0.5;
                  const sinRot = Math.sin(rotAngle);
                  const cosRot = Math.cos(rotAngle);

                  const outL = valL * cosRot - valR * sinRot;
                  const outR = valL * sinRot + valR * cosRot;

                  L[i] = outL * 0.7; 
                  R[i] = outR * 0.7;
              }

              // Update visualization buffer
              if (sCtx.audioVizData) {
                  const captureLen = Math.min(L.length, sCtx.audioVizData.left.length);
                  sCtx.audioVizData.left.set(L.subarray(0, captureLen));
                  sCtx.audioVizData.right.set(R.subarray(0, captureLen));
              }
          };

          processor.connect(gain);
          scriptNodeRef.current = processor;
          setConfig(p => ({...p, audioEnabled: true}));
      } else {
          // Cleanup Audio
          if (audioCtxRef.current) {
              audioCtxRef.current.close();
              audioCtxRef.current = null;
          }
          if (scriptNodeRef.current) {
              scriptNodeRef.current.disconnect();
              scriptNodeRef.current = null;
          }
          setConfig(p => ({...p, audioEnabled: false}));
      }
  };


  // --- Three.js Logic Functions ---

  const rebuildCurve = (ctx: NonNullable<typeof sceneContext.current>) => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < NUM_CTRL; i++) {
      const p = new THREE.Vector3().copy(ctx.ctrlBase[i]).add(ctx.ctrlOffsets[i]);
      pts.push(p);
    }
    ctx.curve = new THREE.CatmullRomCurve3(pts, true, 'catmullrom', ctx.config.tension);
  };

  const updateSamplesAndAttributes = (ctx: NonNullable<typeof sceneContext.current>) => {
    if (!ctx.curve) return;

    // Frenet Frame Calculation
    const frames = ctx.curve.computeFrenetFrames(SAMPLE_N, true);

    for (let i = 0; i < SAMPLE_N; i++) {
      const u = i / SAMPLE_N; 
      
      // Store Points
      ctx.curve.getPointAt(u, ctx.sampled[i]);
      
      // Store Frenet Vectors
      ctx.tangents[i].copy(frames.tangents[i]);
      ctx.normals[i].copy(frames.normals[i]);
      ctx.binormals[i].copy(frames.binormals[i]);
    }

    // Compute Stats (Winding Number)
    let windingAcc = 0;
    for (let i = 0; i < SAMPLE_N; i++) {
      const t = ctx.tangents[i];
      const theta = Math.atan2(t.y, t.x);
      let n = (theta + Math.PI) / (2 * Math.PI); 
      if (n >= 1) n -= 1;
      
      if (i > 0) {
        let d = n - ctx.phaseNorm[i-1];
        if (d > 0.5) d -= 1;
        if (d < -0.5) d += 1;
        windingAcc += d;
      }
      ctx.phaseNorm[i] = n;
    }
    let dLast = ctx.phaseNorm[0] - ctx.phaseNorm[SAMPLE_N - 1];
    if (dLast > 0.5) dLast -= 1;
    if (dLast < -0.5) dLast += 1;
    windingAcc += dLast;
    
    ctx.setStats({ winding: Math.round(windingAcc) });

    // Compute Curvature for Visuals
    const rawK = new Float32Array(SAMPLE_N);
    let kAbsMax = 0;

    for (let i = 0; i < SAMPLE_N; i++) {
        const i0 = (i - 1 + SAMPLE_N) % SAMPLE_N;
        const i2 = (i + 1) % SAMPLE_N;
        const tPrev = ctx.tangents[i0];
        const tNext = ctx.tangents[i2];
        
        const signedTurn = tPrev.x * tNext.y - tPrev.y * tNext.x;
        const dot = THREE.MathUtils.clamp(tPrev.dot(tNext), -1, 1);
        const angle = Math.acos(dot);
        const k = signedTurn >= 0 ? angle : -angle;
        
        rawK[i] = k;
        kAbsMax = Math.max(kAbsMax, Math.abs(k));
    }

    const sampleKAbs: number[] = [];
    for (let i = 0; i < SAMPLE_N; i += 8) sampleKAbs.push(Math.abs(rawK[i]));
    sampleKAbs.sort((a,b) => a - b);
    const q = sampleKAbs[Math.floor(sampleKAbs.length * 0.95)] || (kAbsMax || 1);
    const denom = Math.max(1e-6, q);

    for (let i = 0; i < SAMPLE_N; i++) {
      ctx.curvatureNorm[i] = Math.max(-1, Math.min(1, rawK[i] / denom));
    }

    // Update Geometry
    const positions = ctx.lineGeom.attributes.position.array as Float32Array;
    const colors = ctx.lineGeom.attributes.color.array as Float32Array;
    const { colorMode, viewMode, zScale, showArrows } = ctx.config;

    ctx.arrowGroup.clear();

    for (let i = 0; i < SAMPLE_N; i++) {
        const p = ctx.sampled[i];
        let z = 0;

        if (viewMode === '3d') {
          if (colorMode === 'phase') {
             const theta = (ctx.phaseNorm[i] * 2 * Math.PI) - Math.PI;
             z = (theta / Math.PI) * zScale;
          } else {
             z = ctx.curvatureNorm[i] * zScale;
          }
        }

        positions[i*3+0] = p.x;
        positions[i*3+1] = p.y;
        positions[i*3+2] = z;

        let rgb;
        if (colorMode === 'phase') {
          rgb = hsvToRgb(ctx.phaseNorm[i], 1.0, 1.0);
        } else {
          rgb = rampCurvatureSigned(ctx.curvatureNorm[i]);
        }
        colors[i*3+0] = rgb.r;
        colors[i*3+1] = rgb.g;
        colors[i*3+2] = rgb.b;

        if (showArrows && i % 60 === 0) {
          const origin = new THREE.Vector3(p.x, p.y, z);
          const arrowT = new THREE.ArrowHelper(ctx.tangents[i], origin, 0.15, 0xffffff);
          ctx.arrowGroup.add(arrowT);
        }
    }

    ctx.lineGeom.attributes.position.needsUpdate = true;
    ctx.lineGeom.attributes.color.needsUpdate = true;
    ctx.lineGeom.computeBoundingSphere();
  };

  const syncViewMode = (ctx: NonNullable<typeof sceneContext.current>) => {
     if (ctx.config.viewMode === '2d') {
        ctx.camera.position.set(0, 0, 4.0);
        ctx.controls.target.set(0, 0, 0);
        ctx.controls.minDistance = 1.6;
        ctx.controls.maxDistance = 12;
        ctx.controls.enableRotate = false;
        ctx.grid.visible = false;
     } else {
        ctx.camera.position.set(2.6, 2.1, 2.4);
        ctx.controls.target.set(0, 0, 0.2);
        ctx.controls.minDistance = 1.2;
        ctx.controls.maxDistance = 18;
        ctx.controls.enableRotate = true;
        ctx.grid.visible = true;
     }
     ctx.controls.update();
  };


  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(0, 0, 4.0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x0b0f14, 1);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xdbe7ff, 0x10131a, 0.9);
    scene.add(hemi);

    const grid = new THREE.GridHelper(8, 16, 0x2b3850, 0x172033);
    grid.position.y = 0;
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.22;
    grid.material.transparent = true;
    scene.add(grid);

    const ctrlGroup = new THREE.Group();
    scene.add(ctrlGroup);

    const arrowGroup = new THREE.Group();
    scene.add(arrowGroup);

    const ctrlPoints: THREE.Mesh[] = [];
    const ctrlBase: THREE.Vector3[] = [];
    const ctrlOffsets: THREE.Vector3[] = [];
    const ctrlMat = new THREE.MeshStandardMaterial({ color: 0x0c0c10, roughness: 0.4, metalness: 0.0 });
    const ctrlGeom = new THREE.SphereGeometry(0.045, 24, 16);

    for (let i = 0; i < NUM_CTRL; i++) {
        const u = i / NUM_CTRL;
        const ang = 2 * Math.PI * u;
        const base = new THREE.Vector3(Math.cos(ang) * RADIUS, Math.sin(ang) * RADIUS, 0);
        ctrlBase.push(base);
        ctrlOffsets.push(new THREE.Vector3(0, 0, 0));

        const m = new THREE.Mesh(ctrlGeom, ctrlMat);
        m.position.copy(base);
        m.userData.index = i;
        ctrlGroup.add(m);
        ctrlPoints.push(m);
    }

    // Main Loop Line
    const lineGeom = new THREE.BufferGeometry();
    const posArr = new Float32Array(SAMPLE_N * 3);
    const colArr = new Float32Array(SAMPLE_N * 3);
    lineGeom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    lineGeom.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.98 });
    const line = new THREE.Line(lineGeom, lineMat);
    scene.add(line);

    // Modulator Loop (Trefoil Knot)
    const modGeom = new THREE.BufferGeometry();
    const modPos = new Float32Array(MOD_SAMPLE_N * 3);
    const modSampled: THREE.Vector3[] = [];
    
    // Generate Torus Knot (2,3) for Modulator
    for(let i=0; i<MOD_SAMPLE_N; i++) {
        const t = (i / MOD_SAMPLE_N) * Math.PI * 2;
        // Trefoil / Torus Knot p=2, q=3
        const r = 2.2 + Math.cos(3 * t); // slightly larger radius to surround main loop
        const x = r * Math.cos(2 * t) * 0.6; // Scale down a bit
        const y = r * Math.sin(2 * t) * 0.6;
        const z = Math.sin(3 * t) * 0.5;
        
        modPos[i*3+0] = x;
        modPos[i*3+1] = y;
        modPos[i*3+2] = z;
        modSampled.push(new THREE.Vector3(x, y, z));
    }
    modGeom.setAttribute('position', new THREE.BufferAttribute(modPos, 3));
    const modLine = new THREE.Line(
        modGeom, 
        new THREE.LineBasicMaterial({ color: 0xc084fc, transparent: true, opacity: 0.25 })
    );
    modLine.visible = false;
    scene.add(modLine);

    const particle = new THREE.Mesh(
        new THREE.SphereGeometry(0.055, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0xff2a6d, roughness: 0.35, metalness: 0.0 })
    );
    scene.add(particle);

    const particle2 = new THREE.Mesh(
        new THREE.SphereGeometry(0.055, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0x00e5ff, roughness: 0.2, metalness: 0.1 })
    );
    particle2.visible = false;
    scene.add(particle2);

    const sumParticle = new THREE.Mesh(
        new THREE.SphereGeometry(0.065, 24, 16),
        new THREE.MeshStandardMaterial({ color: 0xffeb3b, emissive: 0xaa8800, emissiveIntensity: 0.2, roughness: 0.2 })
    );
    sumParticle.visible = false;
    scene.add(sumParticle);

    const chordGeom = new THREE.BufferGeometry();
    chordGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const chordLine = new THREE.Line(
        chordGeom,
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 })
    );
    chordLine.visible = false;
    scene.add(chordLine);


    const sampled = new Array(SAMPLE_N).fill(0).map(() => new THREE.Vector3());
    const tangents = new Array(SAMPLE_N).fill(0).map(() => new THREE.Vector3());
    const normals = new Array(SAMPLE_N).fill(0).map(() => new THREE.Vector3());
    const binormals = new Array(SAMPLE_N).fill(0).map(() => new THREE.Vector3());
    const phaseNorm = new Float32Array(SAMPLE_N);
    const curvatureNorm = new Float32Array(SAMPLE_N);

    const ctx = {
        scene, camera, renderer, controls,
        curve: null,
        ctrlPoints, ctrlBase, ctrlOffsets,
        lineGeom, modLine, particle, particle2, sumParticle, chordLine,
        grid, ctrlGroup, arrowGroup,
        sampled, modSampled, tangents, normals, binormals, phaseNorm, curvatureNorm,
        dragging: false,
        dragIndex: -1,
        time: 0,
        config: config,
        setStats: setStats,
        visualChordDistance: 0,
        smoothedDist: 0,
        // Increased buffer for faithful vectorscope
        audioVizData: { left: new Float32Array(2048), right: new Float32Array(2048) }
    };
    sceneContext.current = ctx;

    rebuildCurve(ctx);
    updateSamplesAndAttributes(ctx);
    syncViewMode(ctx);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const dragPlane = new THREE.Plane(new THREE.Vector3(0,0,1), 0);
    const hitPoint = new THREE.Vector3();

    const handlePointerDown = (e: PointerEvent) => {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
        
        raycaster.setFromCamera(pointer, ctx.camera);
        const hits = raycaster.intersectObjects(ctx.ctrlPoints, false);
        
        if (hits.length > 0) {
            ctx.dragging = true;
            ctx.dragIndex = hits[0].object.userData.index;
            ctx.controls.enabled = false;
            renderer.domElement.setPointerCapture(e.pointerId);

            if (ctx.config.viewMode === '3d') {
              const normal = new THREE.Vector3();
              ctx.camera.getWorldDirection(normal);
              dragPlane.setFromNormalAndCoplanarPoint(normal, hits[0].object.position);
            } else {
              dragPlane.set(new THREE.Vector3(0,0,1), 0);
            }
        }
    };

    const handlePointerMove = (e: PointerEvent) => {
        if (!ctx.dragging || ctx.dragIndex < 0) return;

        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);

        raycaster.setFromCamera(pointer, ctx.camera);
        raycaster.ray.intersectPlane(dragPlane, hitPoint);
        
        if (hitPoint) {
           const idx = ctx.dragIndex;
           ctx.ctrlOffsets[idx].subVectors(hitPoint, ctx.ctrlBase[idx]);
           ctx.ctrlPoints[idx].position.copy(ctx.ctrlBase[idx]).add(ctx.ctrlOffsets[idx]);
           rebuildCurve(ctx);
           updateSamplesAndAttributes(ctx);
        }
    };

    const handlePointerUp = (e: PointerEvent) => {
        ctx.dragging = false;
        ctx.dragIndex = -1;
        ctx.controls.enabled = true;
        try { renderer.domElement.releasePointerCapture(e.pointerId); } catch {}
    };

    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);

    let reqId = 0;
    const animate = () => {
        reqId = requestAnimationFrame(animate);
        ctx.controls.update();

        const { isPlaying, speed, colorMode, viewMode, zScale, showDual, showSum, showChord, dualOffset } = ctx.config;

        if (isPlaying) {
            ctx.time += 0.0085 * speed;
        }

        if (ctx.curve) {
            const u1 = (ctx.time % 1 + 1) % 1;
            const p1 = new THREE.Vector3();
            ctx.curve.getPointAt(u1, p1);

            const idx1 = Math.floor(u1 * SAMPLE_N) % SAMPLE_N;
            let z1 = 0;
            if (viewMode === '3d') {
                if (colorMode === 'phase') {
                    const theta = (ctx.phaseNorm[idx1] * 2 * Math.PI) - Math.PI;
                    z1 = (theta / Math.PI) * zScale;
                } else {
                    z1 = ctx.curvatureNorm[idx1] * zScale;
                }
            }
            p1.setZ(z1); 
            ctx.particle.position.copy(p1);

            let rgb;
            if (colorMode === 'phase') {
                rgb = hsvToRgb(ctx.phaseNorm[idx1], 1.0, 1.0);
            } else {
                rgb = rampCurvatureSigned(ctx.curvatureNorm[idx1]);
            }
            ctx.particle.material.color.setRGB(rgb.r, rgb.g, rgb.b);

            const u2 = ((-ctx.time + dualOffset) % 1 + 1) % 1;
            const p2 = new THREE.Vector3();
            ctx.curve.getPointAt(u2, p2);

            const idx2 = Math.floor(u2 * SAMPLE_N) % SAMPLE_N;
            let z2 = 0;
            if (viewMode === '3d') {
                if (colorMode === 'phase') {
                    const theta = (ctx.phaseNorm[idx2] * 2 * Math.PI) - Math.PI;
                    z2 = (theta / Math.PI) * zScale;
                } else {
                    z2 = ctx.curvatureNorm[idx2] * zScale;
                }
            }
            p2.setZ(z2);
            ctx.particle2.position.copy(p2);

            // Sync visual distance for audio LFO
            const dist = p1.distanceTo(p2);
            ctx.visualChordDistance = dist;

            if (showDual && showSum) {
              const sum = new THREE.Vector3().addVectors(p1, p2);
              ctx.sumParticle.position.copy(sum);
            }
            
            if (showDual && showChord) {
              const pos = ctx.chordLine.geometry.attributes.position.array;
              pos[0] = p1.x; pos[1] = p1.y; pos[2] = p1.z;
              pos[3] = p2.x; pos[4] = p2.y; pos[5] = p2.z;
              ctx.chordLine.geometry.attributes.position.needsUpdate = true;
            }
        }

        // --- OVERLAY UPDATES ---

        // 1. Harmonics (Curvature Spread)
        // Average of abs(curvatureNorm)
        let curvSum = 0;
        for(let k=0; k<SAMPLE_N; k+=10) curvSum += Math.abs(ctx.curvatureNorm[k]);
        const avgCurv = (curvSum / (SAMPLE_N/10)); 
        if (harmonicsRef.current) {
            const hPct = Math.min(100, avgCurv * 80);
            harmonicsRef.current.style.width = `${hPct}%`;
            harmonicsRef.current.style.backgroundColor = `hsl(${40 + hPct}, 80%, 60%)`;
        }

        // 2. Phase Variance
        // Ideal step is 1/SAMPLE_N. Measure deviation.
        let varSum = 0;
        for(let k=1; k<SAMPLE_N; k+=10) {
            let d = ctx.phaseNorm[k] - ctx.phaseNorm[k-1];
            if (d < -0.5) d += 1; // wrap
            if (d > 0.5) d -= 1;
            const diff = Math.abs(d - (1/SAMPLE_N));
            varSum += diff;
        }
        if (varianceRef.current) {
            const vVal = Math.min(1, varSum * 100); 
            varianceRef.current.style.opacity = `${0.2 + vVal * 0.8}`;
            varianceRef.current.style.boxShadow = `0 0 ${vVal*20}px ${vVal*5}px rgba(100, 200, 255, ${vVal})`;
        }

        // 3. Vectorscope & Distortion (Canvas)
        const sCanvas = scopeCanvasRef.current;
        const dCanvas = distortCanvasRef.current;

        if (sCanvas && dCanvas) {
            const sCtx = sCanvas.getContext('2d');
            const dCtx = dCanvas.getContext('2d');
            
            if (sCtx && dCtx) {
                // Scope
                sCtx.fillStyle = 'rgba(11, 15, 20, 0.2)'; 
                sCtx.fillRect(0,0, sCanvas.width, sCanvas.height);
                sCtx.fillStyle = '#ffffff';

                // Distortion
                dCtx.clearRect(0,0, dCanvas.width, dCanvas.height);
                dCtx.beginPath();
                dCtx.strokeStyle = '#c084fc';
                dCtx.lineWidth = 2;
                
                // Canvas dims
                const w = sCanvas.width;
                const h = sCanvas.height;
                const dw = dCanvas.width;
                const dh = dCanvas.height;
                
                if (ctx.config.audioEnabled) {
                     // --- REAL AUDIO VISUALIZATION ---
                     const dataL = ctx.audioVizData.left;
                     const dataR = ctx.audioVizData.right;
                     // We use a subset for visualization to keep it snappy
                     const vizLen = 512; 
                     
                     // Draw Waveform (L)
                     dCtx.beginPath();
                     for(let i=0; i<vizLen; i++) {
                        const x = (i/vizLen) * dw;
                        const y = dh/2 + (dataL[i] * dh * 0.45); 
                        if (i===0) dCtx.moveTo(x, y);
                        else dCtx.lineTo(x, y);
                     }
                     dCtx.stroke();
                     
                     // Draw Vectorscope (XY)
                     sCtx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                     // Use a slightly larger sample set for scatter plot if available
                     const scopeLen = Math.min(1024, dataL.length);
                     
                     for(let i=0; i<scopeLen; i+=2) { // step by 2 for performance
                         // L -> X, R -> Y
                         // Scale heavily to see detail
                         const valL = dataL[i];
                         const valR = dataR[i];
                         const x = w/2 + (valL * w * 0.4);
                         const y = h/2 - (valR * h * 0.4);
                         sCtx.fillRect(x, y, 1.2, 1.2);
                     }

                } else {
                    // --- SIMULATION (FALLBACK) ---
                    const { frenetFM, frenetStereo, topoFM, topoSpeed, stereoPhase } = ctx.config;
                    const beta = frenetFM;
                    const dist = ctx.visualChordDistance || 0; 
                    
                    const simN = 180;
                    
                    for(let i=0; i<simN; i++) {
                        const t = (i/simN) * Math.PI * 2;
                        const modPh = t * topoSpeed;
                        const modVal = Math.sin(modPh); 
                        const effPh = t + (topoFM * modVal);
                        
                        // Approximate PM for visual fallback
                        const pMod = effPh + Math.sin(effPh)*beta;
                        
                        const valL = Math.sin(pMod);
                        
                        let phR = pMod;
                        if (stereoPhase > 0.001) phR += (dist * stereoPhase * 0.2);
                        
                        const valR = Math.cos(phR); // 90 deg offset for circle
                        const bzn = Math.sin(t*3) * frenetStereo;
                        
                        // Rotator simulation
                        const rot = bzn * Math.PI * 0.5;
                        const outL = valL * Math.cos(rot) - valR * Math.sin(rot);
                        const outR = valL * Math.sin(rot) + valR * Math.cos(rot);
                        
                        // Draw Distortion (L channel)
                        const dy = dh/2 + (outL * dh * 0.25);
                        if (i===0) dCtx.moveTo(0, dy);
                        else dCtx.lineTo((i/simN)*dw, dy);
                        
                        // Draw Vectorscope
                        const sx = w/2 + (outL * w * 0.22);
                        const sy = h/2 - (outR * h * 0.22); 
                        sCtx.fillRect(sx, sy, 2, 2);
                    }
                    dCtx.stroke();
                }
            }
        }

        renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
        renderer.setSize(window.innerWidth, window.innerHeight);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", handleResize);

    return () => {
        window.removeEventListener("resize", handleResize);
        renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
        renderer.domElement.removeEventListener("pointermove", handlePointerMove);
        renderer.domElement.removeEventListener("pointerup", handlePointerUp);
        cancelAnimationFrame(reqId);
        
        // Audio Cleanup
        if (audioCtxRef.current) audioCtxRef.current.close();
        if (scriptNodeRef.current) scriptNodeRef.current.disconnect();

        renderer.dispose();
        sceneContext.current = null;
        if (mountRef.current) {
            mountRef.current.innerHTML = '';
        }
    };
  }, []); 

  const handleReset = () => {
      if (sceneContext.current) {
          const ctx = sceneContext.current;
          for(let i=0; i<NUM_CTRL; i++) {
              ctx.ctrlOffsets[i].set(0,0,0);
              ctx.ctrlPoints[i].position.copy(ctx.ctrlBase[i]);
          }
          rebuildCurve(ctx);
          updateSamplesAndAttributes(ctx);
      }
  };


  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="absolute inset-0 z-0" />

      {/* Main Controls Panel (Left) */}
      <div className="absolute top-3 left-3 z-10 w-[360px] max-w-[calc(100vw-24px)]">
        <div className="bg-slate-900/80 backdrop-blur-md border border-indigo-500/20 rounded-xl shadow-2xl p-4 text-indigo-50 max-h-[90vh] overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
             <div className="flex items-center gap-3">
               <h1 className="text-sm font-semibold tracking-wide text-white">Loop Visualizer</h1>
               <span className="px-2 py-0.5 rounded-full border border-indigo-400/30 text-[10px] font-medium text-indigo-200 bg-indigo-500/10">WebGL</span>
             </div>
             <div className="text-right">
                <span className="block text-[10px] text-slate-400 uppercase tracking-wider">Winding</span>
                <span className={`text-sm font-mono font-bold ${Math.abs(stats.winding) === 1 ? 'text-green-400' : 'text-amber-400'}`}>
                  {stats.winding.toFixed(0)}
                </span>
             </div>
          </div>

          <div className="space-y-3">
            {/* Visuals Group */}
            <div className="space-y-2 pb-3 border-b border-white/5">
                <div className="flex items-center justify-between">
                <label className="text-xs text-slate-300 w-24">Color Enc.</label>
                <div className="relative flex-1">
                    <select 
                        value={config.colorMode}
                        onChange={(e) => setConfig(prev => ({...prev, colorMode: e.target.value as ColorMode}))}
                        className="w-full bg-white/5 border border-indigo-500/30 rounded-lg py-1.5 px-3 text-xs text-indigo-100 focus:outline-none focus:border-indigo-400"
                    >
                    <option value="phase">Phase θ (direction)</option>
                    <option value="curvature">Curvature κ (signed)</option>
                    </select>
                </div>
                </div>

                <div className="flex items-center justify-between">
                <label className="text-xs text-slate-300 w-24">View</label>
                <div className="flex flex-1 bg-white/5 rounded-lg p-1 border border-indigo-500/20">
                    <button
                        onClick={() => setConfig(p => ({...p, viewMode: '2d'}))}
                        className={`flex-1 py-1 text-[10px] rounded transition-colors ${config.viewMode === '2d' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                        2D
                    </button>
                    <button
                        onClick={() => setConfig(p => ({...p, viewMode: '3d'}))}
                        className={`flex-1 py-1 text-[10px] rounded transition-colors ${config.viewMode === '3d' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                        3D
                    </button>
                </div>
                </div>
            </div>

            {/* Geometry Group */}
            <div className="space-y-2 pb-3 border-b border-white/5">
                <div className="flex items-center justify-between">
                <label className="text-xs text-slate-300 w-24">Z Scale</label>
                <input 
                    type="range" min="0" max="3" step="0.01" 
                    value={config.zScale}
                    onChange={(e) => setConfig(prev => ({...prev, zScale: parseFloat(e.target.value)}))}
                    className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400"
                />
                </div>
                <div className="flex items-center justify-between">
                <label className="text-xs text-slate-300 w-24">Tension</label>
                <input 
                    type="range" min="0" max="1" step="0.01" 
                    value={config.tension}
                    onChange={(e) => setConfig(prev => ({...prev, tension: parseFloat(e.target.value)}))}
                    className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400"
                />
                </div>
            </div>

            {/* Audio Group */}
            <div className="space-y-2 pb-3 border-b border-white/5 bg-emerald-500/5 -mx-4 px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                   <label className="text-xs font-semibold text-emerald-200">Frenet Frame Oscillator</label>
                   <button 
                    onClick={toggleAudio}
                    className={`w-9 h-5 rounded-full relative transition-colors ${config.audioEnabled ? 'bg-emerald-500' : 'bg-slate-700'}`}
                   >
                     <span className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${config.audioEnabled ? 'left-5' : 'left-1'}`} />
                   </button>
                </div>
                
                {config.audioEnabled && (
                  <>
                    <div className="flex items-center justify-between pl-2">
                        <label className="text-[11px] text-slate-400 w-24">Base Freq</label>
                        <input 
                            type="range" min="50" max="800" step="10" 
                            value={config.audioFreq}
                            onChange={(e) => setConfig(prev => ({...prev, audioFreq: parseFloat(e.target.value)}))}
                            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                        <span className="text-[10px] w-8 text-right text-slate-500">{config.audioFreq}</span>
                    </div>
                    <div className="flex items-center justify-between pl-2">
                        <label className="text-[11px] text-slate-400 w-24">Frenet FM</label>
                        <input 
                            type="range" min="0" max="2" step="0.01" 
                            value={config.frenetFM}
                            onChange={(e) => setConfig(prev => ({...prev, frenetFM: parseFloat(e.target.value)}))}
                            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                    </div>
                    <div className="flex items-center justify-between pl-2">
                        <label className="text-[11px] text-slate-400 w-24">Binormal Stereo</label>
                        <input 
                            type="range" min="0" max="1" step="0.01" 
                            value={config.frenetStereo}
                            onChange={(e) => setConfig(prev => ({...prev, frenetStereo: parseFloat(e.target.value)}))}
                            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                    </div>
                  </>
                )}
            </div>

            {/* Topological FM Group */}
            <div className="space-y-2 pb-3 border-b border-white/5 bg-purple-500/5 -mx-4 px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                   <label className="text-xs font-semibold text-purple-200">Topological FM</label>
                   <button 
                    onClick={() => setConfig(p => ({...p, showModulator: !p.showModulator}))}
                    className={`px-2 py-0.5 text-[10px] rounded border ${config.showModulator ? 'bg-purple-500/20 text-purple-200 border-purple-500/30' : 'bg-slate-800 text-slate-500 border-slate-700'}`}
                   >
                     {config.showModulator ? 'Visible' : 'Hidden'}
                   </button>
                </div>
                
                {config.audioEnabled && (
                  <>
                    <div className="flex items-center justify-between pl-2">
                        <label className="text-[11px] text-slate-400 w-24">Topo FM</label>
                        <input 
                            type="range" min="0" max="2" step="0.01" 
                            value={config.topoFM}
                            onChange={(e) => setConfig(prev => ({...prev, topoFM: parseFloat(e.target.value)}))}
                            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />
                    </div>
                    <div className="flex items-center justify-between pl-2">
                        <label className="text-[11px] text-slate-400 w-24">Mod Ratio</label>
                        <input 
                            type="range" min="0.1" max="4" step="0.1" 
                            value={config.topoSpeed}
                            onChange={(e) => setConfig(prev => ({...prev, topoSpeed: parseFloat(e.target.value)}))}
                            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />
                        <span className="text-[10px] w-8 text-right text-slate-500">{config.topoSpeed.toFixed(1)}</span>
                    </div>
                  </>
                )}
            </div>

            {/* Dual Cycle / Summing Feature Group */}
            <div className="space-y-2 pb-3 border-b border-white/5 bg-indigo-500/5 -mx-4 px-4 py-3">
                <div className="flex items-center justify-between mb-2">
                   <label className="text-xs font-semibold text-indigo-200">Dual Cycle (Reverse)</label>
                   <button 
                    onClick={() => setConfig(p => ({...p, showDual: !p.showDual}))}
                    className={`w-9 h-5 rounded-full relative transition-colors ${config.showDual ? 'bg-indigo-500' : 'bg-slate-700'}`}
                   >
                     <span className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${config.showDual ? 'left-5' : 'left-1'}`} />
                   </button>
                </div>
                
                {config.showDual && (
                  <>
                    <div className="flex items-center justify-between pl-2">
                        <label className="text-[11px] text-slate-400 w-24">Phase Offset</label>
                        <input 
                            type="range" min="0" max="1" step="0.01" 
                            value={config.dualOffset}
                            onChange={(e) => setConfig(prev => ({...prev, dualOffset: parseFloat(e.target.value)}))}
                            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                    </div>
                    {config.audioEnabled && (
                      <>
                         <div className="flex items-center justify-between pl-2">
                            <label className="text-[11px] text-slate-400 w-24">Unison Width</label>
                            <input 
                                type="range" min="0" max="2" step="0.01" 
                                value={config.unisonWidth}
                                onChange={(e) => setConfig(prev => ({...prev, unisonWidth: parseFloat(e.target.value)}))}
                                className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                            />
                        </div>
                        <div className="flex items-center justify-between pl-2">
                            <label className="text-[11px] text-slate-400 w-24">Phase Stereo</label>
                            <input 
                                type="range" min="0" max="1" step="0.01" 
                                value={config.stereoPhase}
                                onChange={(e) => setConfig(prev => ({...prev, stereoPhase: parseFloat(e.target.value)}))}
                                className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                            />
                        </div>
                      </>
                    )}
                    <div className="flex gap-2 pl-2 mt-1">
                        <button 
                            onClick={() => setConfig(p => ({...p, showSum: !p.showSum}))}
                            className={`flex-1 py-1.5 text-[10px] border rounded transition-colors ${config.showSum ? 'bg-yellow-500/20 text-yellow-200 border-yellow-500/30' : 'bg-slate-800 text-slate-400 border-slate-700'}`}
                        >
                            Show Sum
                        </button>
                        <button 
                            onClick={() => setConfig(p => ({...p, showChord: !p.showChord}))}
                            className={`flex-1 py-1.5 text-[10px] border rounded transition-colors ${config.showChord ? 'bg-white/10 text-white border-white/20' : 'bg-slate-800 text-slate-400 border-slate-700'}`}
                        >
                            Show Chord
                        </button>
                    </div>
                  </>
                )}
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-3 gap-2 mt-2">
               <button 
                onClick={handleReset}
                className="px-3 py-2 text-xs font-medium text-slate-200 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors"
               >
                 Reset
               </button>
               <button 
                onClick={() => setConfig(p => ({...p, isPlaying: !p.isPlaying}))}
                className={`px-3 py-2 text-xs font-medium border rounded-lg transition-colors ${config.isPlaying ? 'bg-indigo-600/20 text-indigo-200 border-indigo-500/30 hover:bg-indigo-600/30' : 'bg-red-500/10 text-red-200 border-red-500/30 hover:bg-red-500/20'}`}
               >
                 {config.isPlaying ? 'Pause' : 'Play'}
               </button>
               <button 
                onClick={() => setConfig(p => ({...p, showArrows: !p.showArrows}))}
                className={`px-3 py-2 text-xs font-medium bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors ${!config.showArrows ? 'text-slate-500 line-through decoration-slate-500' : 'text-slate-200'}`}
               >
                 {config.showArrows ? 'Hide Arr' : 'Show Arr'}
               </button>
            </div>
          </div>
        </div>
      </div>

      {/* Visualization Overlay (Bottom Right) */}
      <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-3">
          <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/50 rounded-lg p-3 w-[220px] shadow-xl">
             {/* Phase Variance */}
             <div className="flex items-center justify-between mb-1">
                 <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Phase Variance</span>
             </div>
             <div className="w-full h-8 bg-slate-800 rounded mb-3 relative overflow-hidden flex items-center justify-center">
                 <div ref={varianceRef} className="w-4 h-4 rounded-full bg-blue-500 blur-sm transition-opacity" />
             </div>

             {/* Distortion Index */}
             <div className="flex items-center justify-between mb-1">
                 <span className="text-[10px] text-purple-400 uppercase tracking-widest font-semibold">Distortion Idx</span>
             </div>
             <canvas ref={distortCanvasRef} width={200} height={40} className="w-full h-[40px] bg-slate-950 rounded mb-3 border border-purple-500/10" />

             {/* Harmonics Bar */}
             <div className="flex items-center justify-between mb-1">
                 <span className="text-[10px] text-amber-400 uppercase tracking-widest font-semibold">Harmonics</span>
             </div>
             <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden mb-3">
                 <div ref={harmonicsRef} className="h-full bg-amber-500 transition-all duration-100 ease-out w-0" />
             </div>

             {/* Vectorscope */}
             <div className="flex items-center justify-between mb-1">
                 <span className="text-[10px] text-emerald-400 uppercase tracking-widest font-semibold">Stereo Field</span>
             </div>
             <canvas ref={scopeCanvasRef} width={100} height={100} className="w-[100px] h-[100px] mx-auto bg-slate-950 rounded-full border border-emerald-500/20" />
          </div>
      </div>

    </div>
  );
};

export default LoopVisualizer;