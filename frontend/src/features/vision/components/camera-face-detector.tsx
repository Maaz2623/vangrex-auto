"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ============================================================
// TYPES
// ============================================================

type CameraStatus = "idle" | "loading" | "running" | "error";

type EnrolledPerson = {
  id: string;
  name: string;
  samples: string[];
  embeddings?: number[][];
  createdAt: number;
};

type EnrollmentSession = {
  name: string;
  samples: string[];
  embeddings: number[][];
  lastCaptureAt: number;
};

type RecognitionResult = {
  name: string;
  similarity: number;
  recognized: boolean;
};

type HumanFace = {
  box: number[];
  score: number;
  embedding?: number[] | Float32Array;
  mesh?: number[][];
};

type HumanResult = {
  face: HumanFace[];
};

type HumanEngine = {
  load: () => Promise<void>;
  detect: (input: HTMLVideoElement | HTMLImageElement) => Promise<HumanResult>;
  match: {
    similarity: (a: number[], b: number[]) => number;
  };
};

type HumanConstructor = new (config: Record<string, unknown>) => HumanEngine;

declare global {
  interface Window {
    Human?: {
      Human: HumanConstructor;
    };
  }
}

// ============================================================
// CONFIGURATION
// ============================================================

const DATABASE_NAME = "vangrex-home-ai";
const DATABASE_VERSION = 1;
const STORE_NAME = "people";

const HUMAN_VERSION = "3.3.6";

const HUMAN_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/@vladmandic/human@${HUMAN_VERSION}/dist/human.js`;

const MODEL_BASE_PATH = `https://cdn.jsdelivr.net/npm/@vladmandic/human@${HUMAN_VERSION}/models/`;

const REQUIRED_SAMPLES = 5;
const SAMPLE_INTERVAL = 1000;
const MATCH_THRESHOLD = 0.5;

const HUMAN_CONFIG = {
  backend: "webgl",
  modelBasePath: MODEL_BASE_PATH,
  cacheModels: true,
  cacheSensitivity: 0,
  debug: false,
  async: true,
  warmup: "none",

  filter: {
    enabled: false,
  },

  face: {
    enabled: true,

    detector: {
      modelPath: "blazeface.json",
      rotation: true,
      maxDetected: 5,
      minConfidence: 0.45,
      minSize: 40,
      skipFrames: 0,
      skipTime: 0,
      return: true,
    },

    mesh: {
      enabled: true,
      modelPath: "facemesh.json",
    },

    iris: {
      enabled: false,
    },

    attention: {
      enabled: false,
    },

    description: {
      enabled: true,
      modelPath: "faceres.json",
      minConfidence: 0.1,
      skipFrames: 0,
      skipTime: 0,
    },

    emotion: {
      enabled: false,
    },

    antispoof: {
      enabled: false,
    },

    liveness: {
      enabled: false,
    },
  },

  body: {
    enabled: false,
  },

  hand: {
    enabled: false,
  },

  object: {
    enabled: false,
  },

  gesture: {
    enabled: false,
  },

  segmentation: {
    enabled: false,
  },
};

// ============================================================
// BROWSER HUMAN LOADER
// ============================================================

let humanScriptPromise: Promise<HumanConstructor> | null = null;

function loadHumanBrowserBuild(): Promise<HumanConstructor> {
  if (typeof window === "undefined") {
    return Promise.reject(
      new Error("Human can only be loaded in the browser."),
    );
  }

  // The IIFE build exposes a namespace.
  // The actual constructor is window.Human.Human.
  if (window.Human?.Human) {
    return Promise.resolve(window.Human.Human);
  }

  if (humanScriptPromise) {
    return humanScriptPromise;
  }

  humanScriptPromise = new Promise<HumanConstructor>((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>(
      `script[data-vangrex-human="${HUMAN_VERSION}"]`,
    );

    const script = existingScript ?? document.createElement("script");

    const handleLoad = () => {
      const Constructor = window.Human?.Human;

      if (Constructor) {
        resolve(Constructor);
      } else {
        humanScriptPromise = null;

        reject(
          new Error(
            "Human script loaded, but window.Human.Human was not found.",
          ),
        );
      }
    };

    const handleError = () => {
      humanScriptPromise = null;

      reject(new Error("Could not load Human browser library from CDN."));
    };

    script.addEventListener("load", handleLoad, { once: true });
    script.addEventListener("error", handleError, { once: true });

    if (!existingScript) {
      script.src = HUMAN_SCRIPT_URL;
      script.async = true;
      script.dataset.vangrexHuman = HUMAN_VERSION;

      document.head.appendChild(script);
    }
  });

  return humanScriptPromise;
}

// ============================================================
// INDEXEDDB
// ============================================================

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;

      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, {
          keyPath: "id",
        });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getEnrolledPeople(): Promise<EnrolledPerson[]> {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readonly");

    const request = transaction.objectStore(STORE_NAME).getAll();

    request.onsuccess = () => {
      resolve(request.result as EnrolledPerson[]);
      database.close();
    };

    request.onerror = () => {
      reject(request.error);
      database.close();
    };
  });
}

async function saveEnrolledPerson(person: EnrolledPerson): Promise<void> {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");

    transaction.objectStore(STORE_NAME).put(person);

    transaction.oncomplete = () => {
      database.close();
      resolve();
    };

    transaction.onerror = () => {
      database.close();
      reject(transaction.error);
    };

    transaction.onabort = () => {
      database.close();
      reject(transaction.error);
    };
  });
}

async function deleteEnrolledPerson(id: string): Promise<void> {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");

    transaction.objectStore(STORE_NAME).delete(id);

    transaction.oncomplete = () => {
      database.close();
      resolve();
    };

    transaction.onerror = () => {
      database.close();
      reject(transaction.error);
    };
  });
}

// ============================================================
// FACE IMAGE HELPERS
// ============================================================

function captureFaceSample(video: HTMLVideoElement, box: number[]): string {
  const [x, y, width, height] = box;

  const size = Math.max(width, height) * 1.4;

  const centerX = x + width / 2;
  const centerY = y + height / 2;

  const sourceX = Math.max(0, centerX - size / 2);
  const sourceY = Math.max(0, centerY - size / 2);

  const sourceRight = Math.min(video.videoWidth, centerX + size / 2);

  const sourceBottom = Math.min(video.videoHeight, centerY + size / 2);

  const sourceWidth = sourceRight - sourceX;
  const sourceHeight = sourceBottom - sourceY;

  const canvas = document.createElement("canvas");

  canvas.width = 224;
  canvas.height = 224;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Could not create face sample canvas");
  }

  context.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    224,
    224,
  );

  return canvas.toDataURL("image/jpeg", 0.9);
}

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();

    image.onload = () => resolve(image);

    image.onerror = () => {
      reject(new Error("Could not load saved face image"));
    };

    image.src = source;
  });
}

// ============================================================
// MIGRATE OLD ENROLLMENTS
// ============================================================

async function migratePersonEmbeddings(
  human: HumanEngine,
  person: EnrolledPerson,
): Promise<EnrolledPerson> {
  if (person.embeddings?.length) {
    return person;
  }

  const embeddings: number[][] = [];

  for (const sample of person.samples) {
    try {
      const image = await loadImage(sample);
      const result = await human.detect(image);

      const face = [...(result.face ?? [])].sort(
        (a, b) => b.score - a.score,
      )[0];

      if (face?.embedding?.length) {
        embeddings.push(Array.from(face.embedding));
      }
    } catch (error) {
      console.warn(
        `[VANGREX] Could not migrate a sample for ${person.name}:`,
        error,
      );
    }
  }

  const updatedPerson: EnrolledPerson = {
    ...person,
    embeddings,
  };

  await saveEnrolledPerson(updatedPerson);

  return updatedPerson;
}

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function CameraFaceDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const humanRef = useRef<HumanEngine | null>(null);

  const loopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runningRef = useRef(false);
  const lastFrameTimeRef = useRef(0);

  const enrollmentRef = useRef<EnrollmentSession | null>(null);
  const savingRef = useRef(false);

  const peopleRef = useRef<EnrolledPerson[]>([]);

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [message, setMessage] = useState("Camera is stopped");

  const [faceCount, setFaceCount] = useState(0);
  const [fps, setFps] = useState(0);

  const [personName, setPersonName] = useState("");
  const [enrolling, setEnrolling] = useState(false);
  const [enrollmentProgress, setEnrollmentProgress] = useState(0);

  const [enrollmentMessage, setEnrollmentMessage] = useState(
    "Enter a name to enroll a person",
  );

  const [people, setPeople] = useState<EnrolledPerson[]>([]);
  const [databaseLoading, setDatabaseLoading] = useState(true);

  const [recognitionResults, setRecognitionResults] = useState<
    RecognitionResult[]
  >([]);

  const [modelReady, setModelReady] = useState(false);

  const [modelMessage, setModelMessage] = useState(
    "Recognition model not loaded",
  );

  // ==========================================================
  // UPDATE PEOPLE
  // ==========================================================

  const updatePeople = useCallback((next: EnrolledPerson[]) => {
    peopleRef.current = next;
    setPeople(next);
  }, []);

  // ==========================================================
  // LOAD DATABASE
  // ==========================================================

  useEffect(() => {
    let cancelled = false;

    getEnrolledPeople()
      .then((savedPeople) => {
        if (!cancelled) {
          updatePeople(savedPeople);
        }
      })
      .catch((error) => {
        console.error("[VANGREX] Database loading error:", error);
      })
      .finally(() => {
        if (!cancelled) {
          setDatabaseLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [updatePeople]);

  // ==========================================================
  // FACE MATCHING
  // ==========================================================

  const getBestMatch = useCallback((embedding: number[]): RecognitionResult => {
    const human = humanRef.current;

    if (!human) {
      return {
        name: "Unknown",
        similarity: 0,
        recognized: false,
      };
    }

    let bestName = "Unknown";
    let bestSimilarity = 0;

    for (const person of peopleRef.current) {
      for (const savedEmbedding of person.embeddings ?? []) {
        if (savedEmbedding.length !== embedding.length) {
          continue;
        }

        const similarity = human.match.similarity(embedding, savedEmbedding);

        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestName = person.name;
        }
      }
    }

    const recognized = bestSimilarity >= MATCH_THRESHOLD;

    return {
      name: recognized ? bestName : "Unknown",
      similarity: bestSimilarity,
      recognized,
    };
  }, []);

  // ==========================================================
  // STOP CAMERA
  // ==========================================================

  const stopCamera = useCallback(() => {
    runningRef.current = false;

    enrollmentRef.current = null;
    savingRef.current = false;

    if (loopTimerRef.current !== null) {
      clearTimeout(loopTimerRef.current);
      loopTimerRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }

    setFaceCount(0);
    setFps(0);

    setEnrolling(false);
    setEnrollmentProgress(0);

    setRecognitionResults([]);

    setStatus("idle");
    setMessage("Camera is stopped");
  }, []);

  // ==========================================================
  // DETECTION LOOP
  // ==========================================================

  const startDetection = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const human = humanRef.current;

    if (!video || !canvas || !human) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    runningRef.current = true;
    lastFrameTimeRef.current = 0;

    const processFrame = async () => {
      if (!runningRef.current) {
        return;
      }

      if (
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        video.videoWidth === 0
      ) {
        loopTimerRef.current = setTimeout(processFrame, 150);
        return;
      }

      const startedAt = performance.now();

      try {
        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const result = await human.detect(video);

        if (!runningRef.current) {
          return;
        }

        context.clearRect(0, 0, canvas.width, canvas.height);

        const detections = result.face ?? [];

        setFaceCount(detections.length);

        const currentRecognitions: RecognitionResult[] = [];

        for (const face of detections) {
          const [x, y, width, height] = face.box;

          if (width <= 0 || height <= 0) {
            continue;
          }

          let recognition: RecognitionResult = {
            name: "Unknown",
            similarity: 0,
            recognized: false,
          };

          if (face.embedding?.length) {
            recognition = getBestMatch(Array.from(face.embedding));
          }

          currentRecognitions.push(recognition);

          // Draw bounding box.
          context.strokeStyle = recognition.recognized ? "#22c55e" : "#eab308";

          context.lineWidth = Math.max(3, canvas.width / 400);

          context.strokeRect(x, y, width, height);

          // Draw recognition label.
          const confidence = Math.round(recognition.similarity * 100);

          const label = recognition.recognized
            ? `${recognition.name} (${confidence}%)`
            : "Unknown";

          const fontSize = Math.max(18, canvas.width / 65);

          context.font = `bold ${fontSize}px Arial`;

          const textWidth = context.measureText(label).width;

          const labelY = Math.max(0, y - fontSize - 12);

          context.fillStyle = recognition.recognized ? "#16a34a" : "#ca8a04";

          context.fillRect(x, labelY, textWidth + 16, fontSize + 12);

          context.fillStyle = "#ffffff";

          context.fillText(label, x + 8, labelY + fontSize);

          // Draw face mesh points.
          if (face.mesh?.length) {
            context.fillStyle = "#38bdf8";

            for (const point of face.mesh) {
              context.beginPath();

              context.arc(
                point[0],
                point[1],
                Math.max(1.5, canvas.width / 700),
                0,
                Math.PI * 2,
              );

              context.fill();
            }
          }
        }

        setRecognitionResults(currentRecognitions);

        // ====================================================
        // ENROLLMENT
        // ====================================================

        const session = enrollmentRef.current;

        if (session && !savingRef.current) {
          if (detections.length === 0) {
            setEnrollmentMessage("Position your face in front of the camera");
          } else if (detections.length > 1) {
            setEnrollmentMessage("Only one face should be visible");
          } else {
            const face = detections[0];

            if (!face.embedding?.length) {
              setEnrollmentMessage("Waiting for a valid face embedding...");
            } else if (face.box[2] < 80 || face.box[3] < 80) {
              setEnrollmentMessage("Move a little closer to the camera");
            } else if (startedAt - session.lastCaptureAt >= SAMPLE_INTERVAL) {
              const sample = captureFaceSample(video, face.box);

              session.samples.push(sample);

              session.embeddings.push(Array.from(face.embedding));

              session.lastCaptureAt = startedAt;

              const progress = session.samples.length;

              setEnrollmentProgress(progress);

              setEnrollmentMessage(
                `Capturing face samples: ${progress}/${REQUIRED_SAMPLES}`,
              );

              if (progress >= REQUIRED_SAMPLES) {
                enrollmentRef.current = null;
                savingRef.current = true;

                const person: EnrolledPerson = {
                  id: crypto.randomUUID(),
                  name: session.name,
                  samples: session.samples,
                  embeddings: session.embeddings,
                  createdAt: Date.now(),
                };

                void saveEnrolledPerson(person)
                  .then(() => {
                    updatePeople([...peopleRef.current, person]);

                    setEnrollmentMessage(
                      `${person.name} enrolled successfully`,
                    );

                    setPersonName("");
                    setEnrolling(false);
                  })
                  .catch((error) => {
                    console.error("[VANGREX] Enrollment save error:", error);

                    setEnrollmentMessage(
                      "Could not save enrollment. Please try again.",
                    );

                    setEnrolling(false);
                  })
                  .finally(() => {
                    savingRef.current = false;
                  });
              }
            }
          }
        }
      } catch (error) {
        console.error("[VANGREX] Recognition inference error:", error);

        setMessage(
          error instanceof Error
            ? error.message
            : "Face recognition inference failed",
        );
      }

      const finishedAt = performance.now();
      const elapsed = finishedAt - startedAt;

      if (lastFrameTimeRef.current > 0 && elapsed > 0) {
        setFps(Math.round(1000 / elapsed));
      }

      lastFrameTimeRef.current = finishedAt;

      if (runningRef.current) {
        loopTimerRef.current = setTimeout(processFrame, 0);
      }
    };

    void processFrame();
  }, [getBestMatch, updatePeople]);

  // ==========================================================
  // START CAMERA
  // ==========================================================

  const startCamera = useCallback(async () => {
    if (runningRef.current || status === "loading" || databaseLoading) {
      return;
    }

    try {
      setStatus("loading");
      setMessage("Initializing camera...");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      streamRef.current = stream;

      const video = videoRef.current;

      if (!video) {
        throw new Error("Video element not available");
      }

      video.srcObject = stream;

      await video.play();

      setMessage("Loading face recognition AI models...");
      setModelMessage("Loading models...");

      // Load the browser constructor from the IIFE build.
      const Human = await loadHumanBrowserBuild();

      let human = humanRef.current;

      if (!human) {
        human = new Human(HUMAN_CONFIG);
        humanRef.current = human;
      }

      // Load detection, mesh and embedding models.
      await human.load();

      setModelReady(true);
      setModelMessage("Face recognition models ready");

      // Migrate older records that only contain photographs.
      setMessage("Preparing enrolled identities...");

      const migratedPeople: EnrolledPerson[] = [];

      for (const person of peopleRef.current) {
        const updatedPerson = await migratePersonEmbeddings(human, person);

        migratedPeople.push(updatedPerson);
      }

      updatePeople(migratedPeople);

      const withoutEmbeddings = migratedPeople.filter(
        (person) => !person.embeddings?.length,
      );

      if (withoutEmbeddings.length > 0) {
        console.warn(
          "[VANGREX] Some existing records need fresh enrollment:",
          withoutEmbeddings.map((person) => person.name),
        );
      }

      setStatus("running");
      setMessage("Face recognition is active");

      startDetection();
    } catch (error) {
      console.error(
        "[VANGREX] Camera / recognition initialization error:",
        error,
      );

      stopCamera();

      setStatus("error");

      setMessage(
        error instanceof Error
          ? error.message
          : "Failed to initialize face recognition",
      );

      setModelMessage("Model initialization failed");
    }
  }, [status, databaseLoading, startDetection, stopCamera, updatePeople]);

  // ==========================================================
  // BEGIN ENROLLMENT
  // ==========================================================

  const beginEnrollment = useCallback(() => {
    const name = personName.trim();

    if (!name) {
      setEnrollmentMessage("Please enter a person's name");
      return;
    }

    if (status !== "running") {
      setEnrollmentMessage("Start the camera first");
      return;
    }

    if (faceCount !== 1) {
      setEnrollmentMessage("Make sure exactly one face is visible");
      return;
    }

    const alreadyExists = peopleRef.current.some(
      (person) => person.name.toLowerCase() === name.toLowerCase(),
    );

    if (alreadyExists) {
      setEnrollmentMessage(
        "This person is already enrolled. Delete the old record first.",
      );
      return;
    }

    enrollmentRef.current = {
      name,
      samples: [],
      embeddings: [],
      lastCaptureAt: 0,
    };

    setEnrollmentProgress(0);
    setEnrolling(true);

    setEnrollmentMessage("Keep your face visible and look toward the camera");
  }, [personName, status, faceCount]);

  // ==========================================================
  // CANCEL ENROLLMENT
  // ==========================================================

  const cancelEnrollment = useCallback(() => {
    enrollmentRef.current = null;

    setEnrolling(false);
    setEnrollmentProgress(0);

    setEnrollmentMessage("Enrollment cancelled");
  }, []);

  // ==========================================================
  // DELETE PERSON
  // ==========================================================

  const handleDeletePerson = useCallback(
    async (person: EnrolledPerson) => {
      const confirmed = window.confirm(
        `Delete ${person.name} and all their saved face data?`,
      );

      if (!confirmed) {
        return;
      }

      try {
        await deleteEnrolledPerson(person.id);

        updatePeople(peopleRef.current.filter((item) => item.id !== person.id));

        setEnrollmentMessage(`${person.name} was deleted`);
      } catch (error) {
        console.error("[VANGREX] Delete error:", error);

        setEnrollmentMessage("Could not delete this person");
      }
    },
    [updatePeople],
  );

  // ==========================================================
  // CLEANUP
  // ==========================================================

  useEffect(() => {
    return () => {
      runningRef.current = false;

      if (loopTimerRef.current !== null) {
        clearTimeout(loopTimerRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
    };
  }, []);

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="min-h-screen bg-[#09090b] p-6 text-white md:p-10">
      <div className="mx-auto max-w-6xl space-y-8">
        {/* Header */}

        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-sm font-medium tracking-[0.3em] text-green-400">
              VANGREX
            </div>

            <h1 className="mt-2 text-3xl font-semibold">Home AI</h1>

            <p className="mt-1 text-sm text-zinc-400">
              Face detection and identity recognition
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-zinc-800 px-4 py-2">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "running"
                  ? "animate-pulse bg-green-500"
                  : status === "error"
                    ? "bg-red-500"
                    : "bg-zinc-500"
              }`}
            />

            <span className="text-sm text-zinc-300">
              {status === "running"
                ? "System online"
                : status === "loading"
                  ? "Initializing"
                  : status === "error"
                    ? "System error"
                    : "System offline"}
            </span>
          </div>
        </header>

        {/* Dashboard */}

        <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
          {/* Camera */}

          <section className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900">
            <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
              <div>
                <h2 className="font-medium">Live Camera</h2>

                <p className="text-xs text-zinc-500">
                  Face recognition running locally
                </p>
              </div>

              <span className="text-xs text-zinc-500">1280 × 720 target</span>
            </div>

            <div className="relative aspect-video bg-black">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="absolute inset-0 h-full w-full object-fill"
              />

              <canvas
                ref={canvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
              />

              {status !== "running" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-zinc-950">
                  <div className="text-5xl">◉</div>

                  <p className="text-sm text-zinc-400">{message}</p>
                </div>
              )}

              {status === "running" && (
                <div className="absolute left-4 top-4 rounded-lg bg-black/70 px-3 py-2 text-xs text-green-400">
                  ● LIVE RECOGNITION
                </div>
              )}

              {enrolling && (
                <div className="absolute bottom-4 left-4 right-4 rounded-lg border border-green-500/40 bg-black/80 p-3">
                  <p className="text-sm font-medium text-green-400">
                    Enrolling {personName}
                  </p>

                  <p className="mt-1 text-xs text-zinc-300">
                    {enrollmentMessage}
                  </p>

                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-zinc-700">
                    <div
                      className="h-full rounded-full bg-green-500 transition-all"
                      style={{
                        width: `${
                          (enrollmentProgress / REQUIRED_SAMPLES) * 100
                        }%`,
                      }}
                    />
                  </div>

                  <p className="mt-1 text-right text-xs text-zinc-400">
                    {enrollmentProgress}/{REQUIRED_SAMPLES}
                  </p>
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 p-5">
              <p className="text-sm text-zinc-400">{message}</p>

              <div className="flex gap-3">
                <button
                  onClick={stopCamera}
                  disabled={status === "idle"}
                  className="rounded-lg border border-zinc-700 px-5 py-2.5 text-sm hover:bg-zinc-800 disabled:opacity-40"
                >
                  Stop
                </button>

                <button
                  onClick={startCamera}
                  disabled={
                    status === "loading" ||
                    status === "running" ||
                    databaseLoading
                  }
                  className="rounded-lg bg-green-500 px-5 py-2.5 text-sm font-semibold text-black hover:bg-green-400 disabled:opacity-40"
                >
                  {status === "loading" ? "Starting..." : "Start Camera"}
                </button>
              </div>
            </div>
          </section>

          {/* Statistics */}

          <aside className="space-y-4">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
              <p className="text-sm text-zinc-400">Detected faces</p>

              <p className="mt-3 text-5xl font-semibold">{faceCount}</p>

              <p className="mt-2 text-xs text-zinc-500">Currently visible</p>
            </div>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
              <p className="text-sm text-zinc-400">Recognition FPS</p>

              <p className="mt-3 text-5xl font-semibold">{fps}</p>

              <p className="mt-2 text-xs text-zinc-500">
                Approximate inference rate
              </p>
            </div>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
              <p className="text-sm text-zinc-400">Enrolled people</p>

              <p className="mt-3 text-5xl font-semibold">{people.length}</p>

              <p className="mt-2 text-xs text-zinc-500">Stored locally</p>
            </div>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
              <p className="text-sm text-zinc-400">AI model</p>

              <p
                className={`mt-3 text-lg font-medium ${
                  modelReady ? "text-green-400" : "text-yellow-400"
                }`}
              >
                {modelReady ? "Ready" : "Not loaded"}
              </p>

              <p className="mt-2 text-xs text-zinc-500">{modelMessage}</p>
            </div>
          </aside>
        </div>

        {/* Recognition Results */}

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
          <h2 className="text-lg font-semibold">Recognition Results</h2>

          <p className="mt-1 text-sm text-zinc-400">
            Live identity matches against your enrolled database.
          </p>

          {recognitionResults.length === 0 ? (
            <p className="mt-5 text-sm text-zinc-500">
              No faces currently being recognized.
            </p>
          ) : (
            <div className="mt-5 space-y-3">
              {recognitionResults.map((result, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4"
                >
                  <div>
                    <p
                      className={`font-medium ${
                        result.recognized ? "text-green-400" : "text-yellow-400"
                      }`}
                    >
                      {result.name}
                    </p>

                    <p className="mt-1 text-xs text-zinc-500">
                      Face {index + 1}
                    </p>
                  </div>

                  <div className="text-right">
                    <p className="text-lg font-semibold">
                      {Math.round(result.similarity * 100)}%
                    </p>

                    <p className="text-xs text-zinc-500">Similarity</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Person Enrollment */}

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
          <h2 className="text-lg font-semibold">Person Enrollment</h2>

          <p className="mt-1 text-sm text-zinc-400">
            Add a person's name and create their face identity.
          </p>

          <div className="mt-5 flex flex-col gap-3 sm:flex-row">
            <input
              type="text"
              value={personName}
              onChange={(event) => setPersonName(event.target.value)}
              disabled={enrolling || savingRef.current}
              placeholder="Enter person's name"
              maxLength={60}
              className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-4 py-3 text-sm outline-none placeholder:text-zinc-600 focus:border-green-500 disabled:opacity-50"
            />

            {!enrolling ? (
              <button
                onClick={beginEnrollment}
                disabled={status !== "running" || savingRef.current}
                className="rounded-lg bg-green-500 px-6 py-3 text-sm font-semibold text-black hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Enroll Person
              </button>
            ) : (
              <button
                onClick={cancelEnrollment}
                className="rounded-lg border border-red-500/50 px-6 py-3 text-sm text-red-400 hover:bg-red-500/10"
              >
                Cancel Enrollment
              </button>
            )}
          </div>

          <p className="mt-3 text-sm text-zinc-400">{enrollmentMessage}</p>

          <p className="mt-2 text-xs text-zinc-500">
            Exactly one face must be visible during enrollment. Five face
            embeddings and photographs will be captured.
          </p>
        </section>

        {/* Enrolled People */}

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold">Enrolled People</h2>

              <p className="mt-1 text-sm text-zinc-400">
                Local face identity database.
              </p>
            </div>

            <span className="rounded-full bg-zinc-800 px-3 py-1 text-xs text-zinc-300">
              {people.length} saved
            </span>
          </div>

          {databaseLoading ? (
            <p className="mt-5 text-sm text-zinc-500">Loading database...</p>
          ) : people.length === 0 ? (
            <div className="mt-5 rounded-xl border border-dashed border-zinc-700 p-8 text-center">
              <p className="text-sm text-zinc-400">No people enrolled yet.</p>
            </div>
          ) : (
            <div className="mt-5 space-y-3">
              {people.map((person) => (
                <div
                  key={person.id}
                  className="flex items-center justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4"
                >
                  <div className="min-w-0">
                    <p className="truncate font-medium">{person.name}</p>

                    <p className="mt-1 text-xs text-zinc-500">
                      {person.samples.length} photographs ·{" "}
                      {person.embeddings?.length ?? 0} embeddings
                    </p>

                    <p className="mt-1 text-xs text-zinc-600">
                      {new Date(person.createdAt).toLocaleString()}
                    </p>
                  </div>

                  <button
                    onClick={() => void handleDeletePerson(person)}
                    className="shrink-0 rounded-lg border border-red-500/30 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* System Information */}

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
          <h3 className="font-medium">System Information</h3>

          <div className="mt-4 grid gap-4 text-sm sm:grid-cols-3">
            <div>
              <p className="text-zinc-500">Recognition engine</p>

              <p className="mt-1">Vangrex / Human</p>
            </div>

            <div>
              <p className="text-zinc-500">Embedding model</p>

              <p className="mt-1">HSE FaceRes</p>
            </div>

            <div>
              <p className="text-zinc-500">Execution and storage</p>

              <p className="mt-1">Browser / Local IndexedDB</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
