import {Hands} from '@mediapipe/hands';
import {Camera} from '@mediapipe/camera_utils';
import {HAND_CONNECTIONS} from '@mediapipe/hands';
import {drawConnectors, drawLandmarks} from '@mediapipe/drawing_utils';

// vars
const video = document.getElementById('input_video');
const canvas = document.getElementById('hand_canvas');
const ctx = canvas.getContext('2d');
const startWebcamButton = document.getElementById('start-webcam')
const toggleVideoButton = document.getElementById('toggle-video')
const addTrainingsDataButton = document.getElementById('add-to-training-data-button');
const trainButton = document.getElementById('train-button')
const detectButton = document.getElementById('detect-button')
const saveButton = document.getElementById('save-button')
const testButton = document.getElementById('test-button')

const gestureSelector = document.getElementById('gestures');

let handLandmarks = [];

let trainingData = {};
let testData

// NeuralNetwork init
ml5.setBackend("webgl");

// Layers is hier de hidden layers voor de beoordeling! Extra neurons toevoegen verbeterd de accuracy
const neuralNetwork = ml5.neuralNetwork({
    task: 'classification',
    debug: true,
    layers: [
        { type: 'dense', units: 128, activation: 'relu', inputShape: [63] },
        { type: 'dense', units: 64, activation: 'relu' },
        { type: 'dense', units: 32, activation: 'relu' },
        { type: 'dense', units: 5, activation: 'softmax' }
    ]
});
// Camera var
const camera = new Camera(video, {
    onFrame: async () => {
        await hands.send({image: video});
    },
    width: 640,
    height: 480,
});

// event listeners
// start button
startWebcamButton.addEventListener('click', () => {
    init();
})
// toggle video backdrop button
toggleVideoButton.addEventListener('click', () => {
    if (video.style.display === 'block') {
        video.style.display = 'none';
    } else {
        video.style.display = 'block';
    }
})

testButton.addEventListener('click', () => {
    testTraining();
})

// add current pose to training data
addTrainingsDataButton.addEventListener('click', () => {
    trainButton.disabled = false;
    console.log('adding data:',`${gestureSelector.value}`);
    console.log(normalizeHandData());

    const key = gestureSelector.value;
    const data = normalizeHandData();

    if (!trainingData[key]) {
        trainingData[key] = [];
    }

    trainingData[key].push(data);

    console.log('trainingsData object:', trainingData);

    neuralNetwork.addData(normalizeHandData(), {label: `${gestureSelector.value}`});
});

function splitTrainingData(dataObject, trainRatio = 0.8) {
    const trainData = {};
    const testData = {};

    for (const key in dataObject) {
        if (dataObject.hasOwnProperty(key)) {
            const allSamples = dataObject[key];
            const shuffled = [...allSamples].sort(() => Math.random() - 0.5); // Shuffle randomly

            const splitIndex = Math.floor(shuffled.length * trainRatio);
            trainData[key] = shuffled.slice(0, splitIndex);
            testData[key] = shuffled.slice(splitIndex);
        }
    }

    return { trainData, testData };
}

// start AI Training Model (Takes some time to load)
trainButton.addEventListener('click', () => {
    startTraining();
})

// detect the current pose
detectButton.addEventListener('click', () => {
    detectPose();
})

saveButton.addEventListener('click', () => {
    saveTraining();
})

async function startTraining() {
    let splitData = splitTrainingData(trainingData);

    testData = splitData.testData;
    console.log(testData);

    await addDataSet(splitData.trainData);

    await neuralNetwork.normalizeData();
    neuralNetwork.train({epochs: 50}, () => alert('Done!'));
}

async function detectPose() {
    const results = await neuralNetwork.classify(normalizeHandData())
    console.log(results)
}

async function saveTraining() {
    neuralNetwork.save("model", () => console.log("model was saved!"))
}

function finishedTraining() {
    saveButton.disabled = false;
    alert('done');
}

async function testTraining(){
    const trueLabels = [];
    const predictedLabels = [];

    for (const key in testData) {
        if (testData.hasOwnProperty(key)) {
            const samples = testData[key];
            for (let i = 0; i < samples.length; i++) {
                const sample = samples[i];
                const results = await neuralNetwork.classify(sample)

                const bestPrediction = results.reduce((prev, current) =>
                    current.confidence > prev.confidence ? current : prev
                ).label;

                trueLabels.push(key);
                predictedLabels.push(bestPrediction);

                if (bestPrediction === key) {
                    console.log(`✅ Correct: ${key}`);
                } else {
                    console.log(`❌ Incorrect: predicted ${bestPrediction}, expected ${key}`);
                }
            }
        }
    }

    const allLabels = [...new Set([...trueLabels, ...predictedLabels])];

    generateConfusionMatrix(trueLabels, predictedLabels, allLabels);
}

// init
async function init() {
    await camera.start();
    toggleVideoButton.disabled = false;
    addTrainingsDataButton.disabled = false;
    gestureSelector.disabled = false;
}

// ==== Hands Setup ====
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

// make data ready to be learned
function normalizeHandData() {
    const flatHandArray = [];

    if (handLandmarks[0].length > 0) {
        const wrist = handLandmarks[0][0]; // Wrist is always index 0

        for (let i = 0; i < handLandmarks[0].length; i++) {
            const landmark = handLandmarks[0][i];

            const x = landmark.x - wrist.x;
            const y = landmark.y - wrist.y;
            const z = landmark.z - wrist.z;

            flatHandArray.push(x, y, z);
        }
        return flatHandArray;
    }
    return null;
}

async function addDataSet(trainData){
    for (const key in trainData) {
        if (trainData.hasOwnProperty(key)) {
            const samples = trainData[key];
            for (let i = 0; i < samples.length; i++) {
                const sample = samples[i];
                neuralNetwork.addData(sample, { label: `${key}` });
            }
        }
    }
}

// Draw loop
function drawLandmarksFunc() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const hand of handLandmarks) {
        drawConnectors(ctx, hand, HAND_CONNECTIONS, {
            color: 'blue',
            lineWidth: 2
        });

        drawLandmarks(ctx, hand, {
            color: 'green',
            lineWidth: 1
        });
    }
}
// Hands callback
hands.onResults((results) => {
    handLandmarks = results.multiHandLandmarks || [];
    drawLandmarksFunc();
});

function generateConfusionMatrix(trueLabels, predictedLabels, classLabels) {
    const matrix = {};

    classLabels.forEach(actual => {
        matrix[actual] = {};
        classLabels.forEach(predicted => {
            matrix[actual][predicted] = 0;
        });
    });

    for (let i = 0; i < trueLabels.length; i++) {
        const actual = trueLabels[i];
        const predicted = predictedLabels[i];
        matrix[actual][predicted]++;
    }

    console.table(matrix);
}