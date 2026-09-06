/**
 * NutriVision AI - Clean Frontend Application Logic
 * Supports Localhost Testing (Port 5000) & AWS Serverless API Gateway
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let currentImageBase64 = null;
    let selectedPortionScale = 1.0;
    let currentApiUrl = "https://juv9jodvpc.execute-api.ap-southeast-1.amazonaws.com/prod/predict";
    let isSimulatedMode = false;
    let macroChartInstance = null;

    // --- DOM Elements ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const dropContent = document.getElementById('dropContent');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const spinner = document.getElementById('spinner');
    const apiUrlInput = document.getElementById('apiUrl');
    const modeToggleBtn = document.getElementById('modeToggle');
    
    // UI Results Elements
    const placeholderState = document.getElementById('placeholderState');
    const resultState = document.getElementById('resultState');
    const warningBanner = document.getElementById('warningBanner');
    const warningText = document.getElementById('warningText');
    const latencyBadge = document.getElementById('latencyBadge');
    
    const foodNameEl = document.getElementById('foodName');
    const foodCategoryEl = document.getElementById('foodCategory');
    const calorieValueEl = document.getElementById('calorieValue');
    const confidenceTextEl = document.getElementById('confidenceText');
    const confidenceFillEl = document.getElementById('confidenceFill');
    const proteinValEl = document.getElementById('proteinVal');
    const carbsValEl = document.getElementById('carbsVal');
    const fatValEl = document.getElementById('fatVal');
    const fiberValEl = document.getElementById('fiberVal');
    const aiEngineEl = document.getElementById('aiEngine');
    const traceLatencyEl = document.getElementById('traceLatency');
    const traceStatusEl = document.getElementById('traceStatus');

    // Architecture Modal Elements
    const archModal = document.getElementById('archModal');
    const archModalBtn = document.getElementById('archModalBtn');
    const closeModalBtn = document.getElementById('closeModalBtn');

    // --- Portion Selector Pills ---
    const portionBtns = document.querySelectorAll('.portion-btn');
    portionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            portionBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedPortionScale = parseFloat(btn.getAttribute('data-scale')) || 1.0;
            
            if (currentImageBase64) {
                runAnalysis();
            }
        });
    });

    // --- File Drag & Drop Handling ---
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    // Quick Sample Food Pills
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleType = btn.getAttribute('data-sample');
            loadSampleImage(sampleType);
        });
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            showWarning('Vui lòng chọn tệp hình ảnh hợp lệ (JPG, PNG, WebP).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageBase64 = e.target.result;
            imagePreview.src = currentImageBase64;
            imagePreview.classList.remove('hidden');
            dropContent.classList.add('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function loadSampleImage(type) {
        if (type === 'pho') {
            fetch('Pho.jpg')
                .then(res => {
                    if (!res.ok) throw new Error('Could not fetch Pho.jpg');
                    return res.blob();
                })
                .then(blob => {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        currentImageBase64 = e.target.result;
                        imagePreview.src = currentImageBase64;
                        imagePreview.classList.remove('hidden');
                        dropContent.classList.add('hidden');
                        analyzeBtn.disabled = false;
                        runAnalysis();
                    };
                    reader.readAsDataURL(blob);
                })
                .catch(err => {
                    console.warn("Could not load Pho.jpg via fetch, using Image object fallback:", err);
                    const img = new Image();
                    img.onload = () => {
                        const canvas = document.createElement('canvas');
                        canvas.width = img.width || 400;
                        canvas.height = img.height || 400;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        currentImageBase64 = canvas.toDataURL('image/jpeg');
                        imagePreview.src = currentImageBase64;
                        imagePreview.classList.remove('hidden');
                        dropContent.classList.add('hidden');
                        analyzeBtn.disabled = false;
                        runAnalysis();
                    };
                    img.src = 'Pho.jpg';
                });
        }
    }

    // --- Mode Toggle Button ---
    if (apiUrlInput) {
        currentApiUrl = apiUrlInput.value.trim() || "https://juv9jodvpc.execute-api.ap-southeast-1.amazonaws.com/prod/predict";
        apiUrlInput.addEventListener('input', () => {
            currentApiUrl = apiUrlInput.value.trim() || "https://juv9jodvpc.execute-api.ap-southeast-1.amazonaws.com/prod/predict";
        });
    }

    if (modeToggleBtn) {
        modeToggleBtn.textContent = "AWS Cloud Live (API Gateway)";
        modeToggleBtn.addEventListener('click', () => {
            isSimulatedMode = !isSimulatedMode;
            if (isSimulatedMode) {
                modeToggleBtn.textContent = "Localhost Server (Port 5000)";
                currentApiUrl = "http://127.0.0.1:5000/predict";
            } else {
                modeToggleBtn.textContent = "AWS Cloud Live (API Gateway)";
                currentApiUrl = "https://juv9jodvpc.execute-api.ap-southeast-1.amazonaws.com/prod/predict";
            }
            if (apiUrlInput) apiUrlInput.value = currentApiUrl;
        });
    }

    // --- Run Analysis Button Handler ---
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', runAnalysis);
    }

    async function runAnalysis() {
        if (!currentImageBase64) return;

        setLoadingState(true);
        hideWarning();

        const startTime = performance.now();

        try {
            const payload = {
                image: currentImageBase64,
                portion_scale: selectedPortionScale
            };

            const response = await fetch(currentApiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const elapsed = Math.round(performance.now() - startTime);
            const data = await response.json();

            if (response.status === 400) {
                showWarning(`Lỗi Payload (HTTP 400): ${data.message || 'Mã hóa ảnh bị hỏng'}`);
                if (traceStatusEl) traceStatusEl.textContent = "400 BAD REQUEST";
            } else if (response.status === 422 || data.status === 'warning') {
                showWarning(`Cảnh báo chất lượng ảnh (HTTP 422): ${data.message || 'Ảnh quá mờ hoặc quá tối'}`);
                if (traceStatusEl) traceStatusEl.textContent = "422 UNPROCESSABLE ENTITY";
            } else if (data.status === 'low_confidence_warning') {
                showWarning(`Cảnh báo: ${data.warning_message}`);
                renderResults(data, elapsed);
            } else if (response.ok && data.status === 'success') {
                renderResults(data, elapsed);
            } else {
                showWarning(`Lỗi từ server (${response.status}): ${data.message || 'Lỗi không xác định'}`);
            }

        } catch (error) {
            console.error("API Call Error:", error);
            showWarning(`Không thể kết nối tới Server (${currentApiUrl}). Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau.`);
        } finally {
            setLoadingState(false);
        }
    }

    function renderResults(data, elapsed) {
        placeholderState.classList.add('hidden');
        resultState.classList.remove('hidden');

        if (foodNameEl) foodNameEl.textContent = data.name || data.food_class || "Món ăn";
        if (foodCategoryEl) foodCategoryEl.textContent = data.category || "Món Ăn";
        if (calorieValueEl) calorieValueEl.textContent = data.estimated_calories !== undefined && data.estimated_calories !== null ? data.estimated_calories : "--";

        const confPct = ((data.confidence || 0.94) * 100).toFixed(1);
        if (confidenceTextEl) confidenceTextEl.textContent = `${confPct}%`;
        if (confidenceFillEl) confidenceFillEl.style.width = `${confPct}%`;

        const macros = data.macronutrients || { protein_g: 0, carbs_g: 0, fat_g: 0, fiber_g: 0 };
        if (proteinValEl) proteinValEl.textContent = `${macros.protein_g}g`;
        if (carbsValEl) carbsValEl.textContent = `${macros.carbs_g}g`;
        if (fatValEl) fatValEl.textContent = `${macros.fat_g}g`;
        if (fiberValEl) fiberValEl.textContent = `${macros.fiber_g || 3}g`;

        if (latencyBadge) latencyBadge.textContent = `Response: ${elapsed} ms`;
        if (aiEngineEl) aiEngineEl.textContent = data.metadata?.inference_engine || "EfficientNet-B0 Fine-Tuned ONNX Model";
        if (traceLatencyEl) traceLatencyEl.textContent = `${elapsed} ms`;
        if (traceStatusEl) traceStatusEl.textContent = "200 OK";

        renderMacroChart(macros);
    }

    function renderMacroChart(macros) {
        const ctx = document.getElementById('macroChart');
        if (!ctx) return;

        if (macroChartInstance) {
            macroChartInstance.destroy();
        }

        macroChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Protein', 'Carbs', 'Fat', 'Fiber'],
                datasets: [{
                    data: [macros.protein_g, macros.carbs_g, macros.fat_g, macros.fiber_g || 3],
                    backgroundColor: ['#ef4444', '#eab308', '#f97316', '#22c55e'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                cutout: '72%'
            }
        });
    }

    function showWarning(msg) {
        if (warningBanner && warningText) {
            warningText.textContent = msg;
            warningBanner.classList.remove('hidden');
        }
    }

    function hideWarning() {
        if (warningBanner) warningBanner.classList.add('hidden');
    }

    function setLoadingState(isLoading) {
        if (analyzeBtn) analyzeBtn.disabled = isLoading;
        if (spinner) spinner.classList.toggle('hidden', !isLoading);
    }

    // --- Modal Handlers ---
    if (archModalBtn && archModal) {
        archModalBtn.addEventListener('click', () => {
            archModal.classList.remove('hidden');
        });
    }

    if (closeModalBtn && archModal) {
        closeModalBtn.addEventListener('click', () => {
            archModal.classList.add('hidden');
        });
    }

    if (archModal) {
        archModal.addEventListener('click', (e) => {
            if (e.target === archModal) {
                archModal.classList.add('hidden');
            }
        });
    }
});
