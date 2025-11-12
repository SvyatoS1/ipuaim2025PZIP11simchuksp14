// --- 1. КОНФІГУРАЦІЯ ---
const CLASS_NAMES = [
    'graph_1_linear',
    'graph_2_parabola',
    'graph_3_sine'
];
const IMAGES_PER_CLASS = 1000; 
const IMAGE_SIZE = 224; 
const BATCH_SIZE = 32;

const logElement = document.getElementById('log');

// --- 2. ДОПОМІЖНІ ФУНКЦІЇ ---

function log(message) {
    console.log(message);
    if (logElement) {
        logElement.innerHTML += `${message}<br>`;
        logElement.scrollTop = logElement.scrollHeight;
    }
}

async function loadMobilenet() {
    log('Завантаження базової моделі (MobileNetV2)...');
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'
    );
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    const truncatedModel = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });
    for (const layer of truncatedModel.layers) {
        layer.trainable = false;
    }
    log('Базова модель завантажена і "заморожена".');
    return truncatedModel;
}

function buildHeadModel(inputShape) {
    log('Побудова "голови" моделі...');
    const modelHead = tf.sequential();
    modelHead.add(tf.layers.flatten({ inputShape: inputShape.slice(1) }));
    modelHead.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    modelHead.add(tf.layers.dense({
        units: CLASS_NAMES.length,
        activation: 'softmax'
    }));
    log('Модель "голова" побудована.');
    return modelHead;
}

function loadImageToTensor(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        // img.crossOrigin = 'anonymous'; // <--- ЗАКОМЕНТУВАЛИ!
        img.src = url;
        img.onload = () => {
            // Конвертуємо <img> в тензор
            const imgTensor = tf.browser.fromPixels(img);
            
            // 2. Змінюємо розмір до 224x224
            const resizedTensor = tf.image.resizeBilinear(imgTensor, [IMAGE_SIZE, IMAGE_SIZE]);
            
            // 3. Нормалізуємо пікселі
            const normalizedTensor = resizedTensor.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1.0));
            
            // Очищуємо пам'ять
            imgTensor.dispose();
            resizedTensor.dispose();
            
            resolve(normalizedTensor);
        };
        img.onerror = (err) => {
            reject(`Не вдалося завантажити: ${url}`);
        };
    });
}

async function* dataGenerator() {
    log('Створення списку файлів...');
    const allImagePaths = [];
    for (let i = 0; i < CLASS_NAMES.length; i++) {
        const className = CLASS_NAMES[i];
        for (let j = 1; j <= IMAGES_PER_CLASS; j++) {
            const path = `../dataset/${className}/${className}_${j}.png`;
            allImagePaths.push({ path: path, label: i });
        }
    }
    tf.util.shuffle(allImagePaths);
    log(`Знайдено ${allImagePaths.length} зображень. Починаю генератор.`);

    let index = 0;
    while (index < allImagePaths.length) {
        
        // --- ПОЧАТОК ЗМІН ---
        // Ми більше не збираємо проміси. Ми збираємо фактичні тензори.
        const imageTensors = [];
        const batchLabels = [];
        let batchFailed = false;

        // Збираємо дані для одного батчу
        const pathsForBatch = [];
        const labelsForBatch = [];
        for (let i = 0; i < BATCH_SIZE && index < allImagePaths.length; i++) {
            pathsForBatch.push(allImagePaths[index].path);
            labelsForBatch.push(allImagePaths[index].label);
            index++;
        }

        // ТЕПЕР ЗАВАНТАЖУЄМО ЇХ ПО ОДНОМУ
        for (let i = 0; i < pathsForBatch.length; i++) {
            try {
                // 'await' ТУТ змушує код чекати,
                // поки ОДИН файл не завантажиться, перш ніж йти далі.
                const tensor = await loadImageToTensor(pathsForBatch[i]);
                imageTensors.push(tensor);
                batchLabels.push(labelsForBatch[i]);
                
            } catch (err) {
                // Ця помилка тепер означає, що файл ДІЙСНО не знайдено,
                // а не те, що сервер "зайнятий".
                log(`ПОМИЛКА (файл не знайдено?): ${err}. Пропускаю цей батч.`);
                
                // Очищуємо те, що вже завантажили в цьому батчі
                imageTensors.forEach(t => t.dispose());
                batchFailed = true;
                break; // Негайно припиняємо обробку цього батчу
            }
        }

        // Якщо батч провалився (через 1 битий файл), переходимо до наступного
        if (batchFailed) {
            continue;
        }

        // Якщо батч порожній (рідкісний випадок), пропускаємо
        if (imageTensors.length === 0) {
            continue;
        }

        // Успіх! У нас є повний батч, завантажений послідовно
        const xs = tf.stack(imageTensors);
        const ys = tf.oneHot(tf.tensor1d(batchLabels, 'int32'), CLASS_NAMES.length);
        
        // Очищуємо проміжні тензори
        imageTensors.forEach(t => t.dispose());
        
        yield { xs, ys };
        // --- КІНЕЦЬ ЗМІН ---
    }
}

// --- 3. ГОЛОВНА ФУНКЦІЯ ТРЕНУВАННЯ ---

async function runTraining() {
    log('--- Початок тренування ---');
    try {
        const baseModel = await loadMobilenet();
        const headModel = buildHeadModel(baseModel.outputShape);
        const model = tf.model({ inputs: baseModel.inputs, outputs: headModel.apply(baseModel.outputs) });
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        const trainData = tf.data.generator(dataGenerator);
        
        log('Навчання... Це може зайняти кілька хвилин.');
        
        await model.fitDataset(trainData, {
            epochs: 10,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
        // Додаємо перевірку, чи існують 'logs' та 'logs.acc'
                const acc = logs && logs.acc ? logs.acc.toFixed(4) : 'N/A (дані пропущено)';
                const loss = logs && logs.loss ? logs.loss.toFixed(4) : 'N/A';
        
                log(`Епоха ${epoch + 1}/10 - Втрати: ${loss}, Точність: ${acc}`);
                }
            }
        });
        
        log('--- Тренування завершено! ---');
        log('Збереження моделі...');
        
        // ВАЖЛИВО: Зберігаємо модель. Це ініціює завантаження файлів у браузері.
        await model.save('downloads://my-graph-model');
        
        log('Модель збережено!');
        log('Ви повинні побачити 2 файли: my-graph-model.json та my-graph-model.weights.bin.');
        log('Створіть папку "model" у вашому проекті та покладіть обидва файли туди.');

    } catch (err) {
        log(`ПОМИЛКА: ${err}`);
        console.error(err);
    }
}

// --- 4. ЗАПУСК ---
runTraining();