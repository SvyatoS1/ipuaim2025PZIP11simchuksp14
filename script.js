const dragArea = document.querySelector('.drag-area');
const dragText = document.querySelector('.header');
const startButton = document.querySelector('.start-button');
let button = document.querySelector('.button');
let input = document.querySelector('input');

let file;
let hasFile = false;

let model, metadata;
const MODEL_URL = './model/';

const resultText = document.getElementById('result-text');
const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

startButton.disabled = true;
startButton.classList.add('disabled');
resultText.textContent = 'Завантаження моделі...'; 

async function loadModel() {
    const modelURL = MODEL_URL + 'model.json';
    const metadataURL = MODEL_URL + 'metadata.json';
    
    const loadStartTime = performance.now();
    
    try {
        model = await tmImage.load(modelURL, metadataURL);
        metadata = model.getMetadata();
        
        const loadEndTime = performance.now();
        const loadDuration = (loadEndTime - loadStartTime) / 1000;
        
        console.log(`Модель успішно завантажено за ${loadDuration.toFixed(2)}с`, metadata);
        resultText.textContent = `Модель завантажено (${loadDuration.toFixed(2)}с). Оберіть файл.`;
        
        if (hasFile) {
            enableStartButton();
        }
    } catch (err) {
        console.error('Не вдалося завантажити модель:', err);
        resultText.textContent = 'Помилка завантаження моделі. Перезавантажте сторінку.';
    }
}

button.onclick = () => {
    input.click();
};

input.addEventListener('change', function() {
    file = this.files[0];
    dragArea.classList.add('active');
    displayFile();
});

dragArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dragText.textContent = 'Release to upload';
    dragArea.classList.add('active');
});

dragArea.addEventListener('dragleave', () => {
    dragText.textContent = 'Drag & Drop';
    dragArea.classList.remove('active');
});

dragArea.addEventListener('drop', (event) => {
    event.preventDefault();
    file = event.dataTransfer.files[0];
    displayFile();
});

function displayFile() {
    let fileType = file.type;
    let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];

    if (validExtensions.includes(fileType)) {
        let fileReader = new FileReader();

        fileReader.onload = () => {
            let fileURL = fileReader.result;
            let imgTag = `<img src="${fileURL}" alt="Uploaded image" id="uploaded-image-preview">`;
            dragArea.innerHTML = imgTag;

            hasFile = true;
            if (model) {
                enableStartButton();
                resultText.textContent = "Модель готова.";
            } else {
                resultText.textContent = "Файл обрано, очікуємо на завантаження моделі...";
            }
        };

        fileReader.readAsDataURL(file);
    } else {
        alert('This file is not an image');
        dragArea.classList.remove('active');
        hasFile = false;
        disableStartButton();
    }
}

function enableStartButton() {
    startButton.disabled = false;
    startButton.classList.remove('disabled');
}

function disableStartButton() {
    startButton.disabled = true;
    startButton.classList.add('disabled');
}

startButton.addEventListener('click', () => {
    if (Notification.permission === "default") {
        Notification.requestPermission();
    }
    startAnalysis();
});

async function startAnalysis() {
    if (!model || !hasFile) {
        resultText.textContent = 'Помилка: Модель або файл не готові.';
        return;
    }

    const imageElement = document.getElementById('uploaded-image-preview');
    if (!imageElement) {
        resultText.textContent = 'Помилка: Не вдалося знайти зображення.';
        return;
    }

    progressSection.style.display = 'flex';
    progressFill.style.width = '0%';
    progressFill.setAttribute('aria-valuenow', '0');
    progressText.textContent = '0%';
    resultText.textContent = 'Аналіз...';
    resultText.classList.add('active');
    progressFill.style.width = '50%';
    progressText.textContent = '50%';

    try {
        const predictStartTime = performance.now();

        const prediction = await model.predict(imageElement);
        
        const predictEndTime = performance.now();
        const predictDuration = predictEndTime - predictStartTime; 
        
        console.log(`Час прогнозування: ${predictDuration.toFixed(0)} мс`);
        
        
        prediction.sort((a, b) => b.probability - a.probability);

        const resultStrings = prediction.map(p => {
            let cleanName = p.className.split('_').pop(); 
            cleanName = cleanName.charAt(0).toUpperCase() + cleanName.slice(1);
            const confidence = Math.round(p.probability * 100);
            return `${cleanName}: ${confidence}%`;
        });
        
        const fullResultString = resultStrings.join(' | ');
        

        progressFill.style.width = '100%';
        progressText.textContent = '100%';
        
        setTimeout(() => {
            progressSection.style.display = 'none';
            resultText.textContent = `Результат: ${fullResultString} | Аналіз: ${predictDuration.toFixed(0)} мс`;
            showNotification();
        }, 500);

    } catch (err) {
        console.error('Помилка під час прогнозу:', err);
        resultText.textContent = 'Під час аналізу сталася помилка.';
        progressSection.style.display = 'none';
    }
}

function showNotification() {
    if (Notification.permission === 'granted') {
        new Notification("The result of analysis of picture is ready!");
    }
}

loadModel();