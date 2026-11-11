/**
 * VetVision AI — Advanced Veterinary Image Analytics
 * Production Client-Side Engine for 7-Page Multi-Modal Platform
 */

document.addEventListener('DOMContentLoaded', () => {

    // ══════════════════════════════════════════════════════════════
    // ── 1. GLOBAL UTILITIES: TOASTS, CONFIRMATION & THEME ──
    // ══════════════════════════════════════════════════════════════

    function showToast(message, type = 'info', duration = 3500) {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-msg">${message}</span>
        `;

        container.appendChild(toast);

        requestAnimationFrame(() => {
            toast.classList.add('toast-visible');
        });

        setTimeout(() => {
            toast.classList.remove('toast-visible');
            toast.classList.add('toast-hiding');
            setTimeout(() => toast.remove(), 350);
        }, duration);
    }
    window.showToast = showToast;

    // Confirmation Modal
    const confirmModal = document.getElementById('confirmModal');
    const confirmModalTitle = document.getElementById('confirmModalTitle');
    const confirmModalDesc = document.getElementById('confirmModalDesc');
    const confirmModalCancel = document.getElementById('confirmModalCancel');
    const confirmModalConfirm = document.getElementById('confirmModalConfirm');
    let confirmCallback = null;

    function showConfirm(title, description, confirmText, callback) {
        if (!confirmModal) {
            if (confirm(description)) callback();
            return;
        }
        confirmModalTitle.textContent = title;
        confirmModalDesc.textContent = description;
        confirmModalConfirm.textContent = confirmText;
        confirmCallback = callback;
        confirmModal.classList.add('active');
    }
    window.showConfirm = showConfirm;

    function hideConfirm() {
        if (confirmModal) {
            confirmModal.classList.remove('active');
            confirmCallback = null;
        }
    }

    if (confirmModalCancel) confirmModalCancel.addEventListener('click', hideConfirm);
    if (confirmModal) {
        confirmModal.addEventListener('click', (e) => {
            if (e.target === confirmModal) hideConfirm();
        });
    }
    if (confirmModalConfirm) {
        confirmModalConfirm.addEventListener('click', () => {
            if (confirmCallback) confirmCallback();
            hideConfirm();
        });
    }

    // Theme (Dark / Light)
    const themeToggle = document.getElementById('themeToggle');

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('vetvision-theme', theme);
        localStorage.setItem('anivision-theme', theme);
    }

    function getPreferredTheme() {
        const urlParams = new URLSearchParams(window.location.search);
        const urlTheme = urlParams.get('theme');
        if (urlTheme === 'dark' || urlTheme === 'light') return urlTheme;

        const saved = localStorage.getItem('vetvision-theme') || localStorage.getItem('anivision-theme');
        if (saved) return saved;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    setTheme(getPreferredTheme());

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            setTheme(next);
        });
    }

    // Mobile Navigation Drawer
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('mobile-open');
        });
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('mobile-open');
            });
        });
    }

    // Escape Key Accessibility
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideConfirm();
            const diseaseModal = document.getElementById('diseaseModal');
            if (diseaseModal) diseaseModal.style.display = 'none';
            const histModal = document.getElementById('historyDetailModal');
            if (histModal) histModal.style.display = 'none';
        }
    });

    // ══════════════════════════════════════════════════════════════
    // ── 2. DIAGNOSTIC STUDIO WORKFLOW (/detect) ──
    // ══════════════════════════════════════════════════════════════

    const studioExists = document.querySelector('.studio-grid');
    if (studioExists) {
        let activeMediaMode = 'single'; // 'single' | 'multi' | 'video' | 'camera'
        let selectedAnimal = 'dog';
        let singleFile = null;
        let multiFiles = [];
        let videoFile = null;
        let cameraStreamTrack = null;
        let currentFacingMode = 'environment';
        let latestPredictionData = null;

        // URL Animal parameter check (e.g. /detect?animal=cat)
        const urlParams = new URLSearchParams(window.location.search);
        const animalParam = urlParams.get('animal');
        if (animalParam && ['dog', 'cat', 'cow'].includes(animalParam.toLowerCase())) {
            selectedAnimal = animalParam.toLowerCase();
        }

        // Animal Card Selection
        const animalCards = document.querySelectorAll('.animal-card');
        animalCards.forEach(card => {
            const a = card.getAttribute('data-animal');
            if (a === selectedAnimal) {
                card.classList.add('selected');
                card.setAttribute('aria-checked', 'true');
            } else {
                card.classList.remove('selected');
                card.setAttribute('aria-checked', 'false');
            }

            card.addEventListener('click', () => {
                const animal = card.getAttribute('data-animal');
                if (!animal) return;
                animalCards.forEach(c => {
                    c.classList.remove('selected');
                    c.setAttribute('aria-checked', 'false');
                });
                card.classList.add('selected');
                card.setAttribute('aria-checked', 'true');
                selectedAnimal = animal;
                showToast(`Selected ${animal.toUpperCase()} specialized model`, 'info', 2000);
            });
        });

        // Media Mode Switcher Tabs
        const modeButtons = document.querySelectorAll('.media-mode-btn');
        const singleContainer = document.getElementById('singleDropzoneContainer');
        const multiContainer = document.getElementById('multiDropzoneContainer');
        const videoContainer = document.getElementById('videoDropzoneContainer');
        const cameraContainer = document.getElementById('cameraDropzoneContainer');
        const subtitleEl = document.getElementById('mediaModeSubtitle');
        const btnAnalyze = document.getElementById('btnAnalyze');
        const btnAnalyzeText = document.getElementById('btnAnalyzeText');

        function setMediaMode(mode) {
            activeMediaMode = mode;
            modeButtons.forEach(btn => {
                const bMode = btn.getAttribute('data-mode');
                if (bMode === mode) {
                    btn.classList.add('active');
                    btn.setAttribute('aria-selected', 'true');
                } else {
                    btn.classList.remove('active');
                    btn.setAttribute('aria-selected', 'false');
                }
            });

            // Stop camera stream if switching away from camera
            if (mode !== 'camera' && cameraStreamTrack) {
                stopCamera();
            }

            // Hide all containers
            if (singleContainer) singleContainer.style.display = 'none';
            if (multiContainer) multiContainer.style.display = 'none';
            if (videoContainer) videoContainer.style.display = 'none';
            if (cameraContainer) cameraContainer.style.display = 'none';

            if (mode === 'single') {
                if (singleContainer) singleContainer.style.display = 'block';
                if (subtitleEl) subtitleEl.textContent = 'Upload a high-resolution close-up of the affected skin, coat, or eye region';
                if (btnAnalyzeText) btnAnalyzeText.textContent = 'Run AI Screening';
                updateAnalyzeButtonState();
            } else if (mode === 'multi') {
                if (multiContainer) multiContainer.style.display = 'block';
                if (subtitleEl) subtitleEl.textContent = 'Upload 2 to 5 photos from different angles (overhead, close-up, margins)';
                if (btnAnalyzeText) btnAnalyzeText.textContent = `Analyze Multi-Angle Batch (${multiFiles.length} images)`;
                updateAnalyzeButtonState();
            } else if (mode === 'video') {
                if (videoContainer) videoContainer.style.display = 'block';
                if (subtitleEl) subtitleEl.textContent = 'Upload an MP4 or WebM video (5–15 seconds) for automated frame extraction';
                if (btnAnalyzeText) btnAnalyzeText.textContent = 'Extract Frames & Screen Video';
                updateAnalyzeButtonState();
            } else if (mode === 'camera') {
                if (cameraContainer) cameraContainer.style.display = 'block';
                if (subtitleEl) subtitleEl.textContent = 'Capture a clear photo using your device webcam or rear smartphone camera';
                if (btnAnalyzeText) btnAnalyzeText.textContent = 'Run AI Screening';
                startCamera();
            }
        }

        modeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.getAttribute('data-mode');
                if (mode) setMediaMode(mode);
            });
        });

        function updateAnalyzeButtonState() {
            if (!btnAnalyze) return;
            if (activeMediaMode === 'single') {
                btnAnalyze.disabled = !singleFile;
            } else if (activeMediaMode === 'multi') {
                btnAnalyze.disabled = multiFiles.length < 2;
                if (btnAnalyzeText) btnAnalyzeText.textContent = `Analyze Multi-Angle Batch (${multiFiles.length} images)`;
            } else if (activeMediaMode === 'video') {
                btnAnalyze.disabled = !videoFile;
            } else if (activeMediaMode === 'camera') {
                btnAnalyze.disabled = !singleFile;
            }
        }

        // Helper: Format Bytes
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // ── Single Image Dropzone ──
        const dropzone = document.getElementById('dropzone');
        const imageInput = document.getElementById('imageInput');
        const singlePreviewCard = document.getElementById('singlePreviewCard');
        const previewImg = document.getElementById('previewImg');
        const fileNameEl = document.getElementById('fileName');
        const fileSizeEl = document.getElementById('fileSize');
        const btnRemoveSingle = document.getElementById('btnRemoveSingle');
        const qualityBadge = document.getElementById('qualityBadge');
        const qualityText = document.getElementById('qualityText');
        const qualityIndicator = document.getElementById('qualityIndicator');

        if (dropzone && imageInput) {
            ['dragenter', 'dragover'].forEach(name => {
                dropzone.addEventListener(name, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.add('drag-active');
                });
            });

            ['dragleave', 'drop'].forEach(name => {
                dropzone.addEventListener(name, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.remove('drag-active');
                });
            });

            dropzone.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files && files.length > 0) {
                    handleSingleFile(files[0]);
                }
            });

            dropzone.addEventListener('click', () => {
                imageInput.click();
            });

            imageInput.addEventListener('change', () => {
                if (imageInput.files && imageInput.files.length > 0) {
                    handleSingleFile(imageInput.files[0]);
                }
            });
        }

        function handleSingleFile(file) {
            const valid = ['image/jpeg', 'image/png', 'image/webp'];
            if (!valid.includes(file.type)) {
                showToast('Please upload a valid JPG, PNG, or WebP image.', 'error');
                return;
            }
            if (file.size > 30 * 1024 * 1024) {
                showToast('Image file size exceeds 30MB limit.', 'error');
                return;
            }

            singleFile = file;
            if (fileNameEl) fileNameEl.textContent = file.name;
            if (fileSizeEl) fileSizeEl.textContent = formatBytes(file.size);

            const reader = new FileReader();
            reader.onload = (e) => {
                if (previewImg) previewImg.src = e.target.result;
                if (dropzone) dropzone.style.display = 'none';
                if (singlePreviewCard) singlePreviewCard.style.display = 'block';

                // Quality Check
                const testImg = new Image();
                testImg.onload = function () {
                    if (qualityBadge && qualityText && qualityIndicator) {
                        qualityBadge.style.display = 'flex';
                        if (this.width >= 224 && this.height >= 224) {
                            qualityIndicator.style.background = '#10b981';
                            qualityText.textContent = `Optimal Quality: ${this.width}×${this.height}px`;
                        } else {
                            qualityIndicator.style.background = '#f59e0b';
                            qualityText.textContent = `Low Resolution (${this.width}×${this.height}px). Model performs best on 224×224px or higher.`;
                        }
                    }
                };
                testImg.src = e.target.result;

                updateAnalyzeButtonState();
                showToast('Image loaded successfully', 'success');
            };
            reader.readAsDataURL(file);
        }

        if (btnRemoveSingle) {
            btnRemoveSingle.addEventListener('click', () => {
                singleFile = null;
                if (imageInput) imageInput.value = '';
                if (singlePreviewCard) singlePreviewCard.style.display = 'none';
                if (dropzone) dropzone.style.display = 'flex';
                if (qualityBadge) qualityBadge.style.display = 'none';
                updateAnalyzeButtonState();
            });
        }

        // ── Multi-Image Dropzone ──
        const multiDropzone = document.getElementById('multiDropzone');
        const multiImageInput = document.getElementById('multiImageInput');
        const multiPreviewContainer = document.getElementById('multiPreviewContainer');
        const multiPreviewGrid = document.getElementById('multiPreviewGrid');
        const multiCounter = document.getElementById('multiCounter');

        if (multiDropzone && multiImageInput) {
            ['dragenter', 'dragover'].forEach(name => {
                multiDropzone.addEventListener(name, (e) => {
                    e.preventDefault();
                    multiDropzone.classList.add('drag-active');
                });
            });

            ['dragleave', 'drop'].forEach(name => {
                multiDropzone.addEventListener(name, (e) => {
                    e.preventDefault();
                    multiDropzone.classList.remove('drag-active');
                });
            });

            multiDropzone.addEventListener('drop', (e) => {
                const files = Array.from(e.dataTransfer.files);
                handleMultiFiles(files);
            });

            multiDropzone.addEventListener('click', () => {
                multiImageInput.click();
            });

            multiImageInput.addEventListener('change', () => {
                const files = Array.from(multiImageInput.files);
                handleMultiFiles(files);
            });
        }

        function handleMultiFiles(files) {
            const valid = ['image/jpeg', 'image/png', 'image/webp'];
            for (let f of files) {
                if (multiFiles.length >= 5) {
                    showToast('Maximum 5 images allowed for multi-angle batching.', 'warning');
                    break;
                }
                if (valid.includes(f.type) && f.size <= 30 * 1024 * 1024) {
                    multiFiles.push(f);
                }
            }
            renderMultiPreview();
        }

        function renderMultiPreview() {
            if (!multiPreviewGrid) return;
            multiPreviewGrid.innerHTML = '';

            if (multiFiles.length === 0) {
                if (multiPreviewContainer) multiPreviewContainer.style.display = 'none';
                updateAnalyzeButtonState();
                return;
            }

            if (multiPreviewContainer) multiPreviewContainer.style.display = 'block';
            if (multiCounter) multiCounter.textContent = `${multiFiles.length} / 5 images`;

            multiFiles.forEach((file, index) => {
                const card = document.createElement('div');
                card.className = 'preview-thumb-card';

                const img = document.createElement('img');
                const reader = new FileReader();
                reader.onload = (e) => { img.src = e.target.result; };
                reader.readAsDataURL(file);

                const removeBtn = document.createElement('button');
                removeBtn.className = 'thumb-remove-btn';
                removeBtn.textContent = '✕';
                removeBtn.title = 'Remove this image';
                removeBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    multiFiles.splice(index, 1);
                    renderMultiPreview();
                });

                const badge = document.createElement('span');
                badge.className = 'thumb-index-badge';
                badge.textContent = `Angle ${index + 1}`;

                const speciesBadge = document.createElement('span');
                speciesBadge.className = 'thumb-species-badge';
                speciesBadge.textContent = 'Verifying...';
                speciesBadge.style.cssText = 'position:absolute;bottom:6px;left:6px;font-size:10px;font-weight:600;padding:2px 6px;border-radius:4px;background:rgba(15,23,42,0.8);color:#fff;backdrop-filter:blur(4px);pointer-events:none;z-index:2;';

                card.appendChild(img);
                card.appendChild(removeBtn);
                card.appendChild(badge);
                card.appendChild(speciesBadge);
                multiPreviewGrid.appendChild(card);

                // Asynchronously verify species on this preview file
                const checkForm = new FormData();
                checkForm.append('file', file);
                checkForm.append('animal_type', selectedAnimal);
                fetch('/api/verify_species', { method: 'POST', body: checkForm })
                    .then(res => res.json())
                    .then(resData => {
                        if (resData.status === 'species_match') {
                            speciesBadge.textContent = `✓ ${resData.detected_species} (${Math.round(resData.confidence)}%)`;
                            speciesBadge.style.background = '#059669';
                        } else if (resData.status === 'species_mismatch') {
                            speciesBadge.textContent = `⚠ ${resData.detected_species} (Mismatch)`;
                            speciesBadge.style.background = '#d97706';
                        } else if (resData.status === 'unsupported_object') {
                            speciesBadge.textContent = '✕ Non-Animal';
                            speciesBadge.style.background = '#dc2626';
                        } else {
                            speciesBadge.textContent = '? Uncertain';
                            speciesBadge.style.background = '#64748b';
                        }
                    })
                    .catch(() => {
                        speciesBadge.style.display = 'none';
                    });
            });

            updateAnalyzeButtonState();
        }

        // ── Video Dropzone ──
        const videoDropzone = document.getElementById('videoDropzone');
        const videoInput = document.getElementById('videoInput');
        const videoPreviewCard = document.getElementById('videoPreviewCard');
        const videoPlayer = document.getElementById('videoPlayer');
        const videoFileName = document.getElementById('videoFileName');
        const videoFileSize = document.getElementById('videoFileSize');
        const btnRemoveVideo = document.getElementById('btnRemoveVideo');

        if (videoDropzone && videoInput) {
            ['dragenter', 'dragover'].forEach(name => {
                videoDropzone.addEventListener(name, (e) => {
                    e.preventDefault();
                    videoDropzone.classList.add('drag-active');
                });
            });

            ['dragleave', 'drop'].forEach(name => {
                videoDropzone.addEventListener(name, (e) => {
                    e.preventDefault();
                    videoDropzone.classList.remove('drag-active');
                });
            });

            videoDropzone.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files && files.length > 0) handleVideoFile(files[0]);
            });

            videoDropzone.addEventListener('click', () => {
                videoInput.click();
            });

            videoInput.addEventListener('change', () => {
                if (videoInput.files && videoInput.files.length > 0) handleVideoFile(videoInput.files[0]);
            });
        }

        function handleVideoFile(file) {
            const valid = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo'];
            if (!valid.includes(file.type) && !file.name.match(/\.(mp4|webm|mov|avi)$/i)) {
                showToast('Please upload a valid MP4, WebM, or MOV video.', 'error');
                return;
            }
            if (file.size > 30 * 1024 * 1024) {
                showToast('Video size exceeds 30MB limit.', 'error');
                return;
            }

            videoFile = file;
            if (videoFileName) videoFileName.textContent = file.name;
            if (videoFileSize) videoFileSize.textContent = formatBytes(file.size);

            if (videoPlayer) {
                videoPlayer.src = URL.createObjectURL(file);
            }
            if (videoDropzone) videoDropzone.style.display = 'none';
            if (videoPreviewCard) videoPreviewCard.style.display = 'block';

            updateAnalyzeButtonState();
            showToast('Video clip loaded', 'success');
        }

        if (btnRemoveVideo) {
            btnRemoveVideo.addEventListener('click', () => {
                videoFile = null;
                if (videoInput) videoInput.value = '';
                if (videoPlayer) {
                    videoPlayer.pause();
                    videoPlayer.src = '';
                }
                if (videoPreviewCard) videoPreviewCard.style.display = 'none';
                if (videoDropzone) videoDropzone.style.display = 'flex';
                updateAnalyzeButtonState();
            });
        }

        // ── Live Camera Workflow ──
        const cameraStream = document.getElementById('cameraStream');
        const cameraCanvas = document.getElementById('cameraCanvas');
        const btnCameraSnap = document.getElementById('btnCameraSnap');
        const btnCameraStop = document.getElementById('btnCameraStop');
        const btnCameraSwitch = document.getElementById('btnCameraSwitch');

        async function startCamera() {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    showToast('Webcam access not supported in this browser.', 'error');
                    setMediaMode('single');
                    return;
                }
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: currentFacingMode, width: { ideal: 1280 }, height: { ideal: 720 } }
                });
                cameraStreamTrack = stream;
                if (cameraStream) {
                    cameraStream.srcObject = stream;
                }
            } catch (err) {
                console.error('Camera access error:', err);
                showToast('Camera permission denied or camera not available.', 'error');
                setMediaMode('single');
            }
        }

        function stopCamera() {
            if (cameraStreamTrack) {
                cameraStreamTrack.getTracks().forEach(t => t.stop());
                cameraStreamTrack = null;
            }
            if (cameraStream) cameraStream.srcObject = null;
        }

        if (btnCameraStop) {
            btnCameraStop.addEventListener('click', () => {
                stopCamera();
                setMediaMode('single');
            });
        }

        if (btnCameraSwitch) {
            btnCameraSwitch.addEventListener('click', () => {
                currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
                stopCamera();
                startCamera();
            });
        }

        if (btnCameraSnap && cameraCanvas && cameraStream) {
            btnCameraSnap.addEventListener('click', () => {
                const width = cameraStream.videoWidth || 640;
                const height = cameraStream.videoHeight || 480;
                cameraCanvas.width = width;
                cameraCanvas.height = height;
                const ctx = cameraCanvas.getContext('2d');
                ctx.drawImage(cameraStream, 0, 0, width, height);

                cameraCanvas.toBlob((blob) => {
                    const snapFile = new File([blob], `camera_snap_${Date.now()}.jpg`, { type: 'image/jpeg' });
                    stopCamera();
                    setMediaMode('single');
                    handleSingleFile(snapFile);
                    showToast('Photo captured from camera', 'success');
                }, 'image/jpeg', 0.95);
            });
        }

        // ── Processing Stepper Animation ──
        const processingContainer = document.getElementById('processingContainer');
        const pipelineSteps = document.querySelectorAll('.pipeline-step');

        function animateStepper(stepIndex) {
            pipelineSteps.forEach((step, idx) => {
                if (idx < stepIndex) {
                    step.classList.remove('active');
                    step.classList.add('completed');
                } else if (idx === stepIndex) {
                    step.classList.add('active');
                    step.classList.remove('completed');
                } else {
                    step.classList.remove('active', 'completed');
                }
            });
        }

        function resetStepper() {
            pipelineSteps.forEach(s => s.classList.remove('active', 'completed'));
        }

        // ── Analyze Action Execution ──
        if (btnAnalyze) {
            btnAnalyze.addEventListener('click', async () => {
                const mismatchBanner = document.getElementById('mismatchBanner');
                if (mismatchBanner) mismatchBanner.style.display = 'none';

                if (processingContainer) processingContainer.style.display = 'block';
                const resultDashboard = document.getElementById('resultDashboard');
                if (resultDashboard) resultDashboard.style.display = 'none';

                resetStepper();
                animateStepper(0);

                const t1 = setTimeout(() => animateStepper(1), 500);
                const t2 = setTimeout(() => animateStepper(2), 1200);
                const t3 = setTimeout(() => animateStepper(3), 1900);

                const formData = new FormData();
                formData.append('animal_type', selectedAnimal);

                let endpoint = '/predict';

                if (activeMediaMode === 'single' || activeMediaMode === 'camera') {
                    if (!singleFile) return;
                    formData.append('file', singleFile);
                    endpoint = '/predict';
                } else if (activeMediaMode === 'multi') {
                    if (multiFiles.length < 2) return;
                    multiFiles.forEach(f => formData.append('files', f));
                    endpoint = '/api/predict/multi';
                } else if (activeMediaMode === 'video') {
                    if (!videoFile) return;
                    formData.append('file', videoFile);
                    endpoint = '/api/predict/video';
                }

                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });

                    clearTimeout(t1);
                    clearTimeout(t2);
                    clearTimeout(t3);

                    let data = null;
                    try {
                        data = await response.json();
                    } catch (parseErr) {
                        data = null;
                    }

                    if (!response.ok) {
                        const errMsg = (data && (data.error || data.message))
                            ? (data.error || data.message)
                            : `Inference failed with status ${response.status}. Please check network or file format.`;

                        if (processingContainer) processingContainer.style.display = 'none';
                        showToast(errMsg, 'error', 5000);

                        if (data && (data.gatekeeper_error || data.status === 'species_mismatch' || data.status === 'species_uncertain' || data.status === 'unsupported_object' || data.status === 'mixed_species')) {
                            renderDetectionResult(data);
                        }
                        return;
                    }

                    animateStepper(4);

                    setTimeout(() => {
                        if (processingContainer) processingContainer.style.display = 'none';
                        renderDetectionResult(data);
                    }, 500);

                } catch (err) {
                    console.error('Prediction failed:', err);
                    clearTimeout(t1);
                    clearTimeout(t2);
                    clearTimeout(t3);
                    if (processingContainer) processingContainer.style.display = 'none';
                    showToast('Connection error. Could not communicate with server.', 'error');
                }
            });
        }

        // ── Render Detection Results ──
        function renderDetectionResult(data) {
            latestPredictionData = data;
            const resultDashboard = document.getElementById('resultDashboard');
            const mismatchBanner = document.getElementById('mismatchBanner');

            // Gatekeeper Species Mismatch, Unsupported Subject, and Uncertainty Handling
            if (data.gatekeeper_error || data.status === 'species_mismatch' || data.status === 'species_uncertain' || data.status === 'unsupported_object' || data.status === 'mixed_species') {
                if (resultDashboard) resultDashboard.style.display = 'none';
                if (mismatchBanner) {
                    mismatchBanner.style.display = 'flex';
                    mismatchBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    const mismatchTitle = document.getElementById('mismatchTitle');
                    const mismatchDesc = document.getElementById('mismatchDesc');
                    const btnMismatchSwitch = document.getElementById('btnMismatchSwitch');

                    let titleText = 'Species Discrepancy Detected';
                    let descText = data.message || 'The Gatekeeper model detected an animal type different from your selection.';
                    let toastMsg = 'Species discrepancy detected by Gatekeeper AI';
                    let toastType = 'warning';

                    if (data.status === 'unsupported_object') {
                        titleText = 'No Supported Animal Detected';
                        descText = data.message || 'No supported animal detected. Please upload a clear photo of a Dog, Cat, or Cow.';
                        toastMsg = 'No supported animal detected';
                        toastType = 'error';
                    } else if (data.status === 'mixed_species') {
                        titleText = 'Mixed Species Detected';
                        descText = data.message || 'Multiple different animal species were detected across media. VetVision AI evaluates one species at a time.';
                        toastMsg = 'Multiple distinct animal species detected';
                        toastType = 'warning';
                    } else if (data.status === 'species_uncertain') {
                        titleText = 'Species Identification Uncertain';
                        descText = data.message || 'Animal detected, but species could not be identified with confidence. Please upload a clearer photo.';
                        toastMsg = 'Species identification uncertain';
                        toastType = 'warning';
                    } else if (data.status === 'species_mismatch') {
                        titleText = 'Species Mismatch Detected';
                        descText = data.message || `You selected '${data.selected_species}', but Gatekeeper detected a ${data.detected_species}.`;
                        toastMsg = `Species mismatch: detected ${data.detected_species}`;
                        toastType = 'warning';
                    }

                    if (mismatchTitle) mismatchTitle.textContent = titleText;
                    if (mismatchDesc) mismatchDesc.textContent = descText;

                    if (btnMismatchSwitch) {
                        if (data.status === 'species_mismatch' && data.detected_species && ['Dog', 'Cat', 'Cow'].includes(data.detected_species)) {
                            btnMismatchSwitch.style.display = 'inline-flex';
                            btnMismatchSwitch.querySelector('span').textContent = `Switch to ${data.detected_species} Model & Re-Analyze`;
                            btnMismatchSwitch.onclick = () => {
                                const target = data.detected_species.toLowerCase();
                                const card = document.querySelector(`.animal-card[data-animal="${target}"]`);
                                if (card) card.click();
                                mismatchBanner.style.display = 'none';
                                setTimeout(() => {
                                    if (btnAnalyze) btnAnalyze.click();
                                }, 300);
                            };
                        } else {
                            btnMismatchSwitch.style.display = 'none';
                        }
                    }
                }
                showToast(data.message || toastMsg, toastType, 4500);
                return;
            } else {
                if (mismatchBanner) mismatchBanner.style.display = 'none';
            }

            if (!resultDashboard) return;
            resultDashboard.style.display = 'block';
            resultDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Species Badge & Meta
            const resSpeciesBadge = document.getElementById('resSpeciesBadge');
            const resTimestamp = document.getElementById('resTimestamp');
            const resDiseaseName = document.getElementById('resDiseaseName');
            const resSeverityBadge = document.getElementById('resSeverityBadge');
            const resContagionBadge = document.getElementById('resContagionBadge');
            const resSummary = document.getElementById('resSummary');
            const resConfidence = document.getElementById('resConfidence');
            const gaugeCircle = document.getElementById('gaugeCircle');
            const resConfidenceDesc = document.getElementById('resConfidenceDesc');

            const animalName = (data.animal || selectedAnimal).toUpperCase();
            if (resSpeciesBadge) resSpeciesBadge.textContent = `${animalName} MODEL VERIFIED`;
            if (resTimestamp) resTimestamp.textContent = `Evaluated at ${new Date().toLocaleTimeString()}`;
            if (resDiseaseName) resDiseaseName.textContent = data.disease || 'Unknown Condition';

            // Confidence & Gauge
            const conf = parseFloat(data.confidence) || 0;
            if (resConfidence) resConfidence.textContent = `${conf.toFixed(1)}%`;

            if (gaugeCircle) {
                const circ = 2 * Math.PI * 50; // ~314.16
                gaugeCircle.style.strokeDasharray = `${circ}`;
                const offset = circ - (conf / 100) * circ;
                gaugeCircle.style.strokeDashoffset = `${offset}`;
                if (conf >= 70) {
                    gaugeCircle.style.stroke = '#10b981'; // Green
                    if (resConfidenceDesc) resConfidenceDesc.textContent = 'High Diagnostic Confidence';
                } else if (conf >= 40) {
                    gaugeCircle.style.stroke = '#f59e0b'; // Amber
                    if (resConfidenceDesc) resConfidenceDesc.textContent = 'Moderate Diagnostic Confidence';
                } else {
                    gaugeCircle.style.stroke = '#ef4444'; // Red
                    if (resConfidenceDesc) resConfidenceDesc.textContent = 'Low Confidence — Veterinary Review Advised';
                }
            }

            // Consensus Banner for Multi-Image and Video
            const consensusBanner = document.getElementById('resConsensusBanner');
            const resConsensusTitle = document.getElementById('resConsensusTitle');
            const resConsensusDesc = document.getElementById('resConsensusDesc');

            if (consensusBanner) {
                if (data.mode === 'multi_image' || data.mode === 'multi' || data.mode === 'video') {
                    consensusBanner.style.display = 'flex';
                    const agreeScore = data.agreement_score !== undefined ? data.agreement_score : 1.0;
                    if (data.has_mixed_species && data.mixed_species_notice) {
                        consensusBanner.className = 'consensus-banner disagreement';
                        if (resConsensusTitle) resConsensusTitle.textContent = `Mixed Species Excluded (${data.images_analyzed}/${data.total_images_submitted} Analyzed)`;
                        if (resConsensusDesc) resConsensusDesc.textContent = data.mixed_species_notice;
                    } else if (agreeScore >= 0.7) {
                        consensusBanner.className = 'consensus-banner unanimous';
                        if (resConsensusTitle) resConsensusTitle.textContent = `High Consensus Agreement (${Math.round(agreeScore * 100)}%)`;
                        if (resConsensusDesc) resConsensusDesc.textContent = data.consensus_status || 'All sampled perspectives converged reliably on this differential.';
                    } else {
                        consensusBanner.className = 'consensus-banner disagreement';
                        if (resConsensusTitle) resConsensusTitle.textContent = `Differing Perspectives Detected (${Math.round(agreeScore * 100)}% agreement)`;
                        if (resConsensusDesc) resConsensusDesc.textContent = (data.disagreement_details && data.disagreement_details.length) ? data.disagreement_details.join(' • ') : (data.consensus_status || 'Varied perspectives suggest multiple potential differentials. Review Top-3 possibilities.');
                    }
                } else {
                    consensusBanner.style.display = 'none';
                }
            }

            // Top-3 Ranked Probabilities
            const top3List = document.getElementById('top3List');
            if (top3List) {
                top3List.innerHTML = '';
                const top3 = data.top3 || [];
                top3.forEach((item, idx) => {
                    const row = document.createElement('div');
                    row.className = `top3-row rank-${idx + 1}`;
                    const val = parseFloat(item.confidence) || 0;
                    const fillW = Math.max(val, 4);

                    const colors = [
                        'linear-gradient(90deg, #059669, #10b981)',
                        'linear-gradient(90deg, #2563eb, #38bdf8)',
                        'linear-gradient(90deg, #7c3aed, #a855f7)'
                    ];

                    row.innerHTML = `
                        <div class="top3-rank-num rank-${idx + 1}">0${idx + 1}</div>
                        <div class="top3-row-details">
                            <div class="top3-row-text">
                                <span>${item.disease}</span>
                                <span class="top3-conf-val">${val}%</span>
                            </div>
                            <div class="top3-track">
                                <div class="top3-fill" style="width: 0%; background: ${colors[idx] || colors[0]};"></div>
                            </div>
                        </div>
                    `;
                    top3List.appendChild(row);

                    setTimeout(() => {
                        const fill = row.querySelector('.top3-fill');
                        if (fill) fill.style.width = `${fillW}%`;
                    }, 80 + idx * 120);
                });
            }

            // 11-Field Clinical Report Cards
            const clinical = data.clinical_info || {};

            if (resSeverityBadge) resSeverityBadge.textContent = clinical.severity_level || 'Clinical Evaluation';
            if (resContagionBadge) resContagionBadge.textContent = clinical.contagiousness || 'Transmission Assessed';
            if (resSummary) resSummary.textContent = data.description || clinical.description || 'Clinical overview is available below.';

            // Helper to populate a bulleted list
            function populateList(listId, items) {
                const el = document.getElementById(listId);
                if (!el) return;
                el.innerHTML = '';
                if (Array.isArray(items) && items.length > 0) {
                    items.forEach(it => {
                        const li = document.createElement('li');
                        li.textContent = it;
                        el.appendChild(li);
                    });
                } else if (typeof items === 'string' && items.trim()) {
                    const li = document.createElement('li');
                    li.textContent = items;
                    el.appendChild(li);
                } else {
                    el.innerHTML = '<li>Consult with your veterinarian for full assessment.</li>';
                }
            }

            // Helper to populate text
            function populateText(textId, text) {
                const el = document.getElementById(textId);
                if (!el) return;
                el.textContent = text || 'Refer to veterinary clinician for specific recommendations.';
            }

            populateList('listEmergency', clinical.emergency_warning_signs);
            populateList('listSymptoms', clinical.symptoms);
            populateText('textCauses', clinical.causes);
            populateText('textAreas', clinical.affected_areas);
            populateText('textTransmission', clinical.contagiousness);
            populateList('listPrecautions', clinical.precautions);
            populateList('listCare', clinical.supportive_care);
            populateText('textConsultation', clinical.when_to_contact_vet);
            populateList('listSimilar', clinical.similar_conditions);
            populateList('listNextSteps', clinical.recommended_next_steps);
            populateText('textDisclaimer', clinical.disclaimer);

            showToast('Screening analysis generated successfully', 'success');
        }
        window.renderDetectionResult = renderDetectionResult;

        // ── Report Action Toolbar Handlers ──
        const btnReset = document.getElementById('btnReset');
        if (btnReset) {
            btnReset.addEventListener('click', () => {
                const resultDashboard = document.getElementById('resultDashboard');
                if (resultDashboard) resultDashboard.style.display = 'none';

                if (activeMediaMode === 'single') {
                    if (btnRemoveSingle) btnRemoveSingle.click();
                } else if (activeMediaMode === 'multi') {
                    multiFiles = [];
                    renderMultiPreview();
                } else if (activeMediaMode === 'video') {
                    if (btnRemoveVideo) btnRemoveVideo.click();
                }

                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }

        const btnDownloadJson = document.getElementById('btnDownloadJson');
        if (btnDownloadJson) {
            btnDownloadJson.addEventListener('click', () => {
                if (!latestPredictionData) {
                    showToast('No active screening data to download', 'warning');
                    return;
                }
                const blob = new Blob([JSON.stringify(latestPredictionData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `vetvision-screening-${Date.now()}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast('JSON report downloaded', 'success');
            });
        }

        const btnPrintReport = document.getElementById('btnPrintReport');
        if (btnPrintReport) {
            btnPrintReport.addEventListener('click', () => {
                window.print();
            });
        }

        const btnShareReport = document.getElementById('btnShareReport');
        if (btnShareReport) {
            btnShareReport.addEventListener('click', async () => {
                if (!latestPredictionData) {
                    showToast('No active screening data to share', 'warning');
                    return;
                }
                const summary = `VetVision AI Screening Report:
Condition: ${latestPredictionData.disease}
Animal: ${latestPredictionData.animal}
Confidence: ${latestPredictionData.confidence}%
Clinical Severity: ${latestPredictionData.clinical_info ? latestPredictionData.clinical_info.severity_level : 'Assessed'}
Generated by VetVision AI Platform.`;

                if (navigator.share) {
                    try {
                        await navigator.share({
                            title: 'VetVision AI Screening Summary',
                            text: summary,
                            url: window.location.href
                        });
                        showToast('Report shared successfully', 'success');
                    } catch (e) {
                        // User cancelled
                    }
                } else {
                    navigator.clipboard.writeText(summary);
                    showToast('Summary copied to clipboard', 'success');
                }
            });
        }

        // Find a Vet Nearby Toggle & Search
        const btnFindVet = document.getElementById('btnFindVet');
        const vetFinderCard = document.getElementById('vetFinderCard');
        if (btnFindVet && vetFinderCard) {
            btnFindVet.addEventListener('click', () => {
                vetFinderCard.style.display = vetFinderCard.style.display === 'none' ? 'block' : 'none';
                if (vetFinderCard.style.display === 'block') {
                    vetFinderCard.scrollIntoView({ behavior: 'smooth' });
                }
            });
        }

        const btnLaunchMapsGeo = document.getElementById('btnLaunchMapsGeo');
        if (btnLaunchMapsGeo) {
            btnLaunchMapsGeo.addEventListener('click', () => {
                if (!navigator.geolocation) {
                    showToast('Geolocation not supported. Enter your city above.', 'warning');
                    return;
                }
                showToast('Retrieving GPS coordinates...', 'info', 2000);
                navigator.geolocation.getCurrentPosition(
                    pos => {
                        const lat = pos.coords.latitude;
                        const lng = pos.coords.longitude;
                        const url = `https://www.google.com/maps/search/veterinary+clinic/@${lat},${lng},14z`;
                        window.open(url, '_blank');
                    },
                    err => {
                        console.error('GPS error:', err);
                        showToast('Unable to get GPS location. Opening general veterinary search.', 'warning');
                        window.open('https://www.google.com/maps/search/veterinary+clinic+near+me', '_blank');
                    },
                    { timeout: 8000 }
                );
            });
        }

        const btnLaunchMapsSearch = document.getElementById('btnLaunchMapsSearch');
        const vetCustomLocation = document.getElementById('vetCustomLocation');
        if (btnLaunchMapsSearch && vetCustomLocation) {
            btnLaunchMapsSearch.addEventListener('click', () => {
                const loc = vetCustomLocation.value.trim();
                const q = loc ? `veterinary+clinic+in+${encodeURIComponent(loc)}` : 'veterinary+clinic+near+me';
                window.open(`https://www.google.com/maps/search/${q}`, '_blank');
            });
        }
    }

    // ══════════════════════════════════════════════════════════════
    // ── 3. DISEASES KNOWLEDGEBASE PAGE (/diseases) ──
    // ══════════════════════════════════════════════════════════════

    const diseaseGrid = document.getElementById('diseaseGrid');
    if (diseaseGrid) {
        let allDiseases = [];
        let activeSpeciesFilter = 'all';

        // Check URL species parameter (e.g. /diseases?species=cat)
        const urlParams = new URLSearchParams(window.location.search);
        const speciesParam = urlParams.get('species');
        if (speciesParam && ['dog', 'cat', 'cow'].includes(speciesParam.toLowerCase())) {
            activeSpeciesFilter = speciesParam.toLowerCase();
        }

        // Filter tabs
        const filterBtns = document.querySelectorAll('.filter-tab-btn');
        filterBtns.forEach(btn => {
            if (btn.getAttribute('data-species') === activeSpeciesFilter) {
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }

            btn.addEventListener('click', () => {
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeSpeciesFilter = btn.getAttribute('data-species');
                filterDiseases();
            });
        });

        // Search input
        const searchInput = document.getElementById('diseaseSearchInput');
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                filterDiseases();
            });
        }

        // Fetch diseases from API
        async function loadDiseases() {
            try {
                const res = await fetch('/api/diseases');
                const data = await res.json();
                if (data.status === 'success' && data.diseases) {
                    if (Array.isArray(data.diseases)) {
                        allDiseases = data.diseases;
                    } else {
                        allDiseases = Object.entries(data.diseases).map(([key, val]) => ({
                            disease_key: key,
                            display_name: val.disease_name || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                            species: Array.isArray(val.species) ? val.species : (typeof val.species === 'string' ? val.species.split(/\s+/) : ['dog']),
                            ...val
                        }));
                    }
                    filterDiseases();
                }
            } catch (err) {
                console.error('Failed to load diseases:', err);
                diseaseGrid.innerHTML = `
                    <div style="grid-column: 1 / -1; text-align: center; padding: 48px 0; color: var(--color-text-muted);">
                        Failed to load disease library. Please refresh the page.
                    </div>
                `;
            }
        }

        function filterDiseases() {
            const query = searchInput ? searchInput.value.toLowerCase().trim() : '';

            const filtered = allDiseases.filter(d => {
                // Species filter
                const speciesArr = Array.isArray(d.species) ? d.species : [d.species || 'dog'];
                const matchesSpecies = activeSpeciesFilter === 'all' || speciesArr.some(s => s && s.toLowerCase().includes(activeSpeciesFilter));
                // Text search
                const nameMatch = (d.display_name || '').toLowerCase().includes(query);
                const descMatch = (d.description || '').toLowerCase().includes(query);
                const symptoms = Array.isArray(d.symptoms) ? d.symptoms : (typeof d.symptoms === 'string' ? [d.symptoms] : []);
                const symptomsMatch = symptoms.some(s => s.toLowerCase().includes(query));

                return matchesSpecies && (!query || nameMatch || descMatch || symptomsMatch);
            });

            renderDiseaseCards(filtered);
        }

        function renderDiseaseCards(diseases) {
            diseaseGrid.innerHTML = '';
            if (diseases.length === 0) {
                diseaseGrid.innerHTML = `
                    <div style="grid-column: 1 / -1; text-align: center; padding: 48px 0;">
                        <p style="font-size: 1.1rem; color: var(--color-text); font-weight: 700;">No conditions matched your filter.</p>
                        <p style="font-size: 0.9rem; color: var(--color-text-muted);">Try searching for a different symptom or select "All Species".</p>
                    </div>
                `;
                return;
            }

            diseases.forEach(d => {
                const card = document.createElement('div');
                card.className = 'disease-catalog-card';

                const speciesList = Array.isArray(d.species) ? d.species : [d.species || 'dog'];
                const primarySpecies = speciesList[0] || 'dog';
                const speciesClass = primarySpecies.toLowerCase();

                card.innerHTML = `
                    <div class="disease-card-top">
                        <span class="species-tag ${speciesClass}">${speciesList.map(s => s.toUpperCase()).join(' • ')}</span>
                        <span class="severity-badge" style="font-size: 0.75rem;">${d.severity_level || 'Clinical'}</span>
                    </div>
                    <h3 class="disease-card-title">${d.display_name}</h3>
                    <p class="disease-card-desc">${d.description || 'Clinical overview and symptom manifestations.'}</p>
                    <div class="disease-card-footer">
                        <span class="disease-card-contagion">${d.contagiousness || 'Contagion assessed'}</span>
                        <span style="font-weight: 700; color: var(--color-primary);">View Guide &rarr;</span>
                    </div>
                `;

                card.addEventListener('click', () => {
                    openDiseaseModal(d);
                });

                diseaseGrid.appendChild(card);
            });
        }

        // Disease Detail Modal
        const diseaseModal = document.getElementById('diseaseModal');
        const modalSpeciesTag = document.getElementById('modalSpeciesTag');
        const modalDiseaseTitle = document.getElementById('modalDiseaseTitle');
        const modalSeverityBadge = document.getElementById('modalSeverityBadge');
        const modalContagionBadge = document.getElementById('modalContagionBadge');
        const modalDiseaseContent = document.getElementById('modalDiseaseContent');
        const btnCloseDiseaseModal = document.getElementById('btnCloseDiseaseModal');
        const btnDismissModal = document.getElementById('btnDismissModal');

        function openDiseaseModal(d) {
            if (!diseaseModal) return;
            const speciesList = Array.isArray(d.species) ? d.species : [d.species || 'dog'];
            const primary = speciesList[0] || 'dog';

            if (modalSpeciesTag) {
                modalSpeciesTag.className = `species-tag ${primary.toLowerCase()}`;
                modalSpeciesTag.textContent = speciesList.map(s => s.toUpperCase()).join(' • ');
            }
            if (modalDiseaseTitle) modalDiseaseTitle.textContent = d.display_name;
            if (modalSeverityBadge) modalSeverityBadge.textContent = d.severity_level || 'Assessed';
            if (modalContagionBadge) modalContagionBadge.textContent = d.contagiousness || 'Transmission Assessed';

            function renderBullets(items) {
                if (Array.isArray(items) && items.length > 0) {
                    return `<ul class="clinical-list">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
                }
                return `<p>${items || 'Consult veterinarian.'}</p>`;
            }

            if (modalDiseaseContent) {
                modalDiseaseContent.innerHTML = `
                    <div style="margin-bottom: 20px;">
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Clinical Overview</h4>
                        <p style="font-size: 0.9rem; color: var(--color-text-muted); line-height: 1.6;">${d.description || ''}</p>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px;">
                        <div class="clinical-card symptoms">
                            <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Clinical Symptoms</h4>
                            ${renderBullets(d.symptoms)}
                        </div>

                        <div class="clinical-card causes">
                            <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Primary Etiology &amp; Causes</h4>
                            <p style="font-size: 0.875rem; color: var(--color-text-muted);">${d.causes || 'Unknown'}</p>
                        </div>

                        <div class="clinical-card">
                            <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Affected Anatomical Areas</h4>
                            <p style="font-size: 0.875rem; color: var(--color-text-muted);">${d.affected_areas || 'Whole body'}</p>
                        </div>

                        <div class="clinical-card transmission">
                            <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Contagiousness</h4>
                            <p style="font-size: 0.875rem; color: var(--color-text-muted);">${d.contagiousness || 'Non-contagious'}</p>
                        </div>

                        <div class="clinical-card care">
                            <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Supportive Care</h4>
                            ${renderBullets(d.supportive_care)}
                        </div>

                        <div class="clinical-card precautions">
                            <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Precautions</h4>
                            ${renderBullets(d.precautions)}
                        </div>
                    </div>

                    <div class="clinical-card emergency" style="margin-bottom: 16px;">
                        <h4 style="font-size: 0.9rem; font-weight: 700; color: #dc2626; margin-bottom: 6px;">🚨 Emergency Warning Signs</h4>
                        ${renderBullets(d.emergency_warning_signs)}
                    </div>

                    <div class="clinical-card consultation">
                        <h4 style="font-size: 0.9rem; font-weight: 700; color: #2563eb; margin-bottom: 6px;">🩺 When to Contact a Veterinarian</h4>
                        <p style="font-size: 0.875rem; color: var(--color-text-muted);">${d.when_to_contact_vet || 'Schedule a professional exam.'}</p>
                    </div>
                `;
            }

            diseaseModal.style.display = 'flex';
        }

        function closeDiseaseModal() {
            if (diseaseModal) diseaseModal.style.display = 'none';
        }

        if (btnCloseDiseaseModal) btnCloseDiseaseModal.addEventListener('click', closeDiseaseModal);
        if (btnDismissModal) btnDismissModal.addEventListener('click', closeDiseaseModal);
        if (diseaseModal) {
            diseaseModal.addEventListener('click', (e) => {
                if (e.target === diseaseModal) closeDiseaseModal();
            });
        }

        loadDiseases();
    }

    // ══════════════════════════════════════════════════════════════
    // ── 4. SCREENING HISTORY DASHBOARD (/history) ──
    // ══════════════════════════════════════════════════════════════

    const historyCardsGrid = document.getElementById('historyCardsGrid');
    if (historyCardsGrid) {
        let historyData = [];
        let activeHistFilter = 'all';

        const filterBtns = document.querySelectorAll('[data-hist-filter]');
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeHistFilter = btn.getAttribute('data-hist-filter');
                filterAndRenderHistory();
            });
        });

        const sortSelect = document.getElementById('historySortSelect');
        if (sortSelect) {
            sortSelect.addEventListener('change', () => {
                filterAndRenderHistory();
            });
        }

        async function fetchHistory() {
            try {
                const res = await fetch('/api/history');
                const data = await res.json();
                if (data.status === 'success' && Array.isArray(data.history)) {
                    historyData = data.history;
                    updateHistoryMetrics();
                    filterAndRenderHistory();
                }
            } catch (err) {
                console.error('Failed to fetch history:', err);
            }
        }

        function updateHistoryMetrics() {
            const total = historyData.length;
            const highConf = historyData.filter(i => (parseFloat(i.confidence) || 0) >= 70).length;
            const reviewNeeded = historyData.filter(i => (parseFloat(i.confidence) || 0) < 50).length;

            const dogs = historyData.filter(i => (i.animal || '').toLowerCase() === 'dog').length;
            const cats = historyData.filter(i => (i.animal || '').toLowerCase() === 'cat').length;
            const cows = historyData.filter(i => (i.animal || '').toLowerCase() === 'cow').length;

            const elTotal = document.getElementById('histStatTotal');
            const elHighConf = document.getElementById('histStatHighConf');
            const elReview = document.getElementById('histStatReview');
            const elSpecies = document.getElementById('histStatSpecies');

            if (elTotal) elTotal.textContent = total;
            if (elHighConf) elHighConf.textContent = highConf;
            if (elReview) elReview.textContent = reviewNeeded;
            if (elSpecies) elSpecies.textContent = `${dogs} / ${cats} / ${cows}`;
        }

        function filterAndRenderHistory() {
            let list = [...historyData];

            // Filter by species
            if (activeHistFilter !== 'all') {
                list = list.filter(i => (i.animal || '').toLowerCase() === activeHistFilter);
            }

            // Sort
            const sortMode = sortSelect ? sortSelect.value : 'newest';
            if (sortMode === 'newest') {
                list.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            } else if (sortMode === 'oldest') {
                list.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            } else if (sortMode === 'conf_desc') {
                list.sort((a, b) => (parseFloat(b.confidence) || 0) - (parseFloat(a.confidence) || 0));
            } else if (sortMode === 'conf_asc') {
                list.sort((a, b) => (parseFloat(a.confidence) || 0) - (parseFloat(b.confidence) || 0));
            }

            renderHistoryCards(list);
        }

        function renderHistoryCards(items) {
            historyCardsGrid.innerHTML = '';
            const emptyState = document.getElementById('historyEmptyState');

            if (items.length === 0) {
                if (emptyState) emptyState.style.display = 'block';
                return;
            }

            if (emptyState) emptyState.style.display = 'none';

            const emojis = { dog: '🐕', cat: '🐈', cow: '🐄' };

            items.forEach(item => {
                const card = document.createElement('div');
                card.className = 'history-card';

                const animalKey = (item.animal || 'dog').toLowerCase();
                const confVal = parseFloat(item.confidence) || 0;
                const confClass = confVal >= 70 ? 'high' : 'low';

                card.innerHTML = `
                    <div class="history-card-img">
                        <img src="${item.image_path || '/static/favicon.svg'}" alt="${item.prediction}" loading="lazy">
                        <span class="history-card-animal-badge">${emojis[animalKey] || '🐾'} ${item.animal}</span>
                    </div>
                    <div class="history-card-body">
                        <h4 class="history-card-disease">${item.prediction}</h4>
                        <div class="history-card-conf conf-${confClass}">
                            <span class="history-conf-bar" style="width: ${Math.max(confVal, 5)}%"></span>
                            <span class="history-conf-text">${confVal}%</span>
                        </div>
                        <div class="history-card-date">${new Date(item.timestamp).toLocaleString()}</div>
                    </div>
                    <div class="history-card-actions">
                        <button class="btn-history-view" data-action="view" data-id="${item.id}">View Report</button>
                        <button class="btn-history-delete" data-action="delete" data-id="${item.id}">Delete</button>
                    </div>
                `;

                card.querySelector('[data-action="view"]').addEventListener('click', () => {
                    openHistoryDetail(item);
                });

                card.querySelector('[data-action="delete"]').addEventListener('click', () => {
                    deleteHistoryItem(item.id);
                });

                historyCardsGrid.appendChild(card);
            });
        }

        // History Detail Modal
        const historyDetailModal = document.getElementById('historyDetailModal');
        const histDetailSpecies = document.getElementById('histDetailSpecies');
        const histDetailDisease = document.getElementById('histDetailDisease');
        const histDetailTimestamp = document.getElementById('histDetailTimestamp');
        const histDetailImg = document.getElementById('histDetailImg');
        const histDetailBody = document.getElementById('histDetailBody');
        const btnCloseHistDetail = document.getElementById('btnCloseHistDetail');
        const btnDismissHistDetail = document.getElementById('btnDismissHistDetail');
        const btnDeleteHistItem = document.getElementById('btnDeleteHistItem');
        let currentDetailId = null;

        function openHistoryDetail(item) {
            if (!historyDetailModal) return;
            currentDetailId = item.id;

            const species = (item.animal || 'Dog').toLowerCase();
            if (histDetailSpecies) {
                histDetailSpecies.className = `species-tag ${species}`;
                histDetailSpecies.textContent = item.animal.toUpperCase();
            }
            if (histDetailDisease) histDetailDisease.textContent = item.prediction;
            if (histDetailTimestamp) histDetailTimestamp.textContent = new Date(item.timestamp).toLocaleString();
            if (histDetailImg) histDetailImg.src = item.image_path || '';

            let top3HTML = '';
            const topPreds = item.top_predictions || [];
            topPreds.forEach((t, i) => {
                top3HTML += `<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--color-border); font-size: 0.85rem;">
                    <span>0${i + 1}. ${t.disease}</span>
                    <strong style="color: var(--color-primary);">${t.confidence}%</strong>
                </div>`;
            });

            if (histDetailBody) {
                histDetailBody.innerHTML = `
                    <div style="margin-bottom: 16px;">
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Differential Rankings</h4>
                        ${top3HTML || '<p style="color: var(--color-text-muted);">No ranked differentials stored.</p>'}
                    </div>
                    <div style="margin-bottom: 16px;">
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Clinical Overview</h4>
                        <p style="font-size: 0.875rem; color: var(--color-text-muted); line-height: 1.6;">${item.description || 'No overview recorded.'}</p>
                    </div>
                    <div>
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: var(--color-text); margin-bottom: 6px;">Veterinary Guidance</h4>
                        <p style="font-size: 0.875rem; color: var(--color-text-muted); line-height: 1.6;">${item.action || 'Consult with a registered veterinarian.'}</p>
                    </div>
                `;
            }

            historyDetailModal.style.display = 'flex';
        }

        function closeHistoryDetail() {
            if (historyDetailModal) historyDetailModal.style.display = 'none';
            currentDetailId = null;
        }

        if (btnCloseHistDetail) btnCloseHistDetail.addEventListener('click', closeHistoryDetail);
        if (btnDismissHistDetail) btnDismissHistDetail.addEventListener('click', closeHistoryDetail);
        if (historyDetailModal) {
            historyDetailModal.addEventListener('click', (e) => {
                if (e.target === historyDetailModal) closeHistoryDetail();
            });
        }

        if (btnDeleteHistItem) {
            btnDeleteHistItem.addEventListener('click', () => {
                if (currentDetailId) {
                    closeHistoryDetail();
                    deleteHistoryItem(currentDetailId);
                }
            });
        }

        // Delete single item
        function deleteHistoryItem(id) {
            showConfirm(
                'Delete Record',
                'Are you sure you want to delete this screening record?',
                'DELETE RECORD',
                async () => {
                    try {
                        const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
                        const data = await res.json();
                        if (data.status === 'success') {
                            showToast('Record deleted successfully', 'success');
                            fetchHistory();
                        } else {
                            showToast('Failed to delete record', 'error');
                        }
                    } catch (e) {
                        showToast('Error deleting record', 'error');
                    }
                }
            );
        }

        // Clear All History
        const btnClearAllHistory = document.getElementById('btnClearAllHistory');
        if (btnClearAllHistory) {
            btnClearAllHistory.addEventListener('click', () => {
                showConfirm(
                    'Clear All History',
                    'Permanently delete all saved veterinary screening records? This cannot be undone.',
                    'CLEAR ALL HISTORY',
                    async () => {
                        try {
                            const res = await fetch('/api/history', { method: 'DELETE' });
                            const data = await res.json();
                            if (data.status === 'success') {
                                showToast('All history records cleared', 'success');
                                fetchHistory();
                            } else {
                                showToast('Failed to clear history', 'error');
                            }
                        } catch (e) {
                            showToast('Error clearing history', 'error');
                        }
                    }
                );
            });
        }

        fetchHistory();
    }

    // ══════════════════════════════════════════════════════════════
    // ── 5. GUIDE PAGE VET FINDER TOOL (/guide) ──
    // ══════════════════════════════════════════════════════════════

    const btnGuideFindVet = document.getElementById('btnGuideFindVet');
    const guideVetLocation = document.getElementById('guideVetLocation');

    if (btnGuideFindVet) {
        btnGuideFindVet.addEventListener('click', () => {
            const loc = guideVetLocation ? guideVetLocation.value.trim() : '';
            if (loc) {
                window.open(`https://www.google.com/maps/search/veterinary+clinic+in+${encodeURIComponent(loc)}`, '_blank');
            } else if (navigator.geolocation) {
                showToast('Retrieving GPS coordinates...', 'info', 2000);
                navigator.geolocation.getCurrentPosition(
                    pos => {
                        const lat = pos.coords.latitude;
                        const lng = pos.coords.longitude;
                        window.open(`https://www.google.com/maps/search/veterinary+clinic/@${lat},${lng},14z`, '_blank');
                    },
                    () => {
                        window.open('https://www.google.com/maps/search/veterinary+clinic+near+me', '_blank');
                    },
                    { timeout: 7000 }
                );
            } else {
                window.open('https://www.google.com/maps/search/veterinary+clinic+near+me', '_blank');
            }
        });
    }

});
