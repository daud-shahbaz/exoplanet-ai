// NASA Exoplanet Detection System - JavaScript

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
    
    // Load visualizations if visualize tab is selected
    if (tabName === 'visualize') {
        loadVisualizations();
    }
}

// Feature ranges for mapping 0-100 slider to real astronomical values
const FEATURE_RANGES = {
    'koi_period': { min: 0.5, max: 700, unit: 'days' },
    'koi_depth': { min: 10, max: 100000, unit: 'ppm' },
    'koi_duration': { min: 0.5, max: 20, unit: 'hours' },
    'koi_prad': { min: 0.5, max: 30, unit: 'Earth radii' },
    'koi_teq': { min: 100, max: 3000, unit: 'K' },
    'koi_insol': { min: 0.01, max: 2000, unit: 'flux' },
    'koi_steff': { min: 3000, max: 8000, unit: 'K' },
    'koi_slogg': { min: 3.5, max: 5.0, unit: '' },
    'koi_srad': { min: 0.5, max: 3.0, unit: 'Solar radii' },
    'koi_impact': { min: 0.0, max: 1.0, unit: '' }
};

// Map slider value (0-100) to astronomical range
function mapSliderToRange(featureName, sliderValue) {
    const range = FEATURE_RANGES[featureName];
    if (!range) return sliderValue;
    
    // Linear mapping from 0-100 to min-max
    const mapped = range.min + (sliderValue / 100) * (range.max - range.min);
    return mapped;
}

// Update slider values with real astronomical values
function updateValue(featureName) {
    const slider = document.getElementById(featureName);
    const display = document.getElementById(`${featureName}-value`);
    const sliderValue = parseFloat(slider.value);
    
    // Map to real value
    const realValue = mapSliderToRange(featureName, sliderValue);
    
    // Display with appropriate precision and unit
    const range = FEATURE_RANGES[featureName];
    let displayText;
    
    if (realValue < 1) {
        displayText = realValue.toFixed(3);
    } else if (realValue < 10) {
        displayText = realValue.toFixed(2);
    } else if (realValue < 100) {
        displayText = realValue.toFixed(1);
    } else {
        displayText = realValue.toFixed(0);
    }
    
    if (range && range.unit) {
        displayText += ' ' + range.unit;
    }
    
    display.textContent = displayText;
    display.title = `Slider: ${sliderValue}/100 → Real value: ${realValue.toFixed(2)}`;
}

// Manual Prediction
async function predictManual() {
    const btn = event.target;
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Detecting...';
    
    // Clear previous warnings
    const oldWarnings = document.querySelector('.validation-warnings');
    if (oldWarnings) oldWarnings.remove();
    
    // Get all feature values and convert slider values to real astronomical values
    const features = Array.from(document.querySelectorAll('.input-group input[type="range"]'));
    const data = {};
    
    features.forEach(input => {
        const sliderValue = parseFloat(input.value);
        // Convert 0-100 slider to real astronomical value
        const realValue = mapSliderToRange(input.name, sliderValue);
        data[input.name] = realValue;
    });
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayManualResult(result.result);
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🚀 Detect Exoplanet';
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <strong>❌ Error:</strong> ${message}
    `;
    errorDiv.style.cssText = `
        background: #ea4335;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        animation: fadeIn 0.3s ease;
    `;
    
    const container = document.querySelector('.tab-content.active');
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => {
        errorDiv.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
}

// Display Manual Prediction Result
function displayManualResult(result) {
    const container = document.getElementById('manual-result');
    container.style.display = 'block';
    
    // Show validation warnings if any
    if (result.validation_warnings && result.validation_warnings.length > 0) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'validation-warnings';
        warningDiv.innerHTML = `
            <h4>⚠️ Input Validation Warnings:</h4>
            <ul>
                ${result.validation_warnings.map(w => `<li>${w}</li>`).join('')}
            </ul>
            <p><em>These values are outside typical astronomical ranges but prediction will proceed.</em></p>
        `;
        container.insertBefore(warningDiv, container.firstChild);
    }
    
    // Prediction
    const predictionEl = document.getElementById('result-prediction');
    predictionEl.textContent = result.prediction;
    predictionEl.style.color = result.probability > 0.5 ? '#34a853' : '#ea4335';
    
    // Probability bar
    const probBar = document.getElementById('probability-bar');
    probBar.style.width = `${result.probability * 100}%`;
    probBar.textContent = `${(result.probability * 100).toFixed(1)}%`;
    
    // Confidence with model agreement
    const confidenceEl = document.getElementById('result-confidence');
    const agreementText = result.model_agreement ? 
        ` | Model Agreement: ${result.model_agreement.toFixed(1)}%` : '';
    confidenceEl.textContent = `Confidence: ${result.confidence.toFixed(1)}%${agreementText}`;
    confidenceEl.title = 'Model agreement shows how much the 3 models agree on this prediction';
    
    // Model scores
    document.getElementById('dnn-bar').style.width = `${result.model_scores.dnn * 100}%`;
    document.getElementById('dnn-score').textContent = `${(result.model_scores.dnn * 100).toFixed(1)}%`;
    
    document.getElementById('cnn-bar').style.width = `${result.model_scores.cnn * 100}%`;
    document.getElementById('cnn-score').textContent = `${(result.model_scores.cnn * 100).toFixed(1)}%`;
    
    document.getElementById('lstm-bar').style.width = `${result.model_scores.lstm * 100}%`;
    document.getElementById('lstm-score').textContent = `${(result.model_scores.lstm * 100).toFixed(1)}%`;
    
    // Feature contributions chart
    displayContributionsChart(result.contributions);
    
    // Scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display Feature Contributions Chart
function displayContributionsChart(contributions) {
    const topFeatures = Object.entries(contributions).slice(0, 10);
    
    const data = [{
        type: 'bar',
        x: topFeatures.map(([name, data]) => data.contribution),
        y: topFeatures.map(([name, _]) => name.replace('koi_', '').replace('_', ' ')),
        orientation: 'h',
        marker: {
            color: topFeatures.map(([_, data]) => data.contribution),
            colorscale: 'Viridis'
        }
    }];
    
    const layout = {
        title: {
            text: 'Top 10 Feature Contributions',
            font: { size: 18, color: '#1e293b', weight: 'bold' }
        },
        xaxis: { 
            title: { text: 'Contribution Score', font: { size: 14, color: '#334155' } },
            tickfont: { size: 12, color: '#475569' },
            gridcolor: '#e2e8f0'
        },
        yaxis: { 
            title: '',
            tickfont: { size: 12, color: '#475569' }
        },
        height: 400,
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f8fafc',
        font: { color: '#1e293b' },
        margin: { l: 120, r: 40, t: 60, b: 60 }
    };
    
    Plotly.newPlot('contributions-chart', data, layout, {responsive: true});
}

// File Upload Handling
function handleFileSelect() {
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadBtn = document.getElementById('upload-btn');
    
    if (fileInput.files.length > 0) {
        fileName.textContent = fileInput.files[0].name;
        uploadBtn.disabled = false;
    } else {
        fileName.textContent = 'Choose CSV file...';
        uploadBtn.disabled = true;
    }
}

// File Upload Prediction
async function predictFile() {
    const fileInput = document.getElementById('file-input');
    const btn = event.target;
    
    if (!fileInput.files.length) {
        showError('Please select a CSV file first');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB');
        return;
    }
    
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Processing ' + file.name + '...';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayBatchResults(result);
            btn.innerHTML = '✓ Analysis Complete!';
            setTimeout(() => {
                btn.innerHTML = '🔬 Analyze Dataset';
            }, 3000);
        } else {
            showError(result.error || 'Analysis failed');
            btn.innerHTML = '🔬 Analyze Dataset';
        }
    } catch (error) {
        showError('Network error: ' + error.message);
        btn.innerHTML = '🔬 Analyze Dataset';
    } finally {
        btn.disabled = false;
    }
}

// Display Batch Results
function displayBatchResults(result) {
    const container = document.getElementById('upload-result');
    container.style.display = 'block';
    
    // Statistics
    document.getElementById('total-count').textContent = result.count;
    document.getElementById('detected-count').textContent = result.detected;
    document.getElementById('detection-rate').textContent = result.detection_rate;
    
    // Add data quality info if available
    if (result.data_quality) {
        const qualityInfo = document.createElement('div');
        qualityInfo.className = 'data-quality-info';
        qualityInfo.innerHTML = `
            <h4>📊 Data Quality</h4>
            <p>Missing values: ${result.data_quality.missing_values} (${result.data_quality.missing_percentage})</p>
            ${result.avg_confidence ? `<p>Average confidence: ${result.avg_confidence}</p>` : ''}
        `;
        qualityInfo.style.cssText = `
            background: rgba(102, 126, 234, 0.1);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        `;
        
        const statsDiv = document.querySelector('.batch-stats');
        statsDiv.after(qualityInfo);
    }
    
    // Results table
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    // Show first 100 results
    const displayResults = result.results.slice(0, 100);
    
    displayResults.forEach((item, idx) => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${item.index + 1}</td>
            <td style="color: ${item.probability > 0.5 ? '#34a853' : '#ea4335'}; font-weight: bold;">
                ${item.prediction}
            </td>
            <td>${(item.probability * 100).toFixed(1)}%</td>
            <td>${item.confidence.toFixed(1)}%</td>
            <td>${(item.model_scores.dnn * 100).toFixed(1)}%</td>
            <td>${(item.model_scores.cnn * 100).toFixed(1)}%</td>
            <td>${(item.model_scores.lstm * 100).toFixed(1)}%</td>
        `;
    });
    
    // Batch chart
    displayBatchChart(result.results);
    
    // Scroll to results
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display Batch Chart
function displayBatchChart(results) {
    const probabilities = results.map(r => r.probability);
    
    const data = [{
        type: 'histogram',
        x: probabilities,
        nbinsx: 20,
        marker: {
            color: '#667eea',
        }
    }];
    
    const layout = {
        title: {
            text: 'Distribution of Exoplanet Probabilities',
            font: { size: 18, color: '#1e293b', weight: 'bold' }
        },
        xaxis: { 
            title: { text: 'Probability', font: { size: 14, color: '#334155' } },
            range: [0, 1],
            tickfont: { size: 12, color: '#475569' },
            gridcolor: '#e2e8f0'
        },
        yaxis: { 
            title: { text: 'Count', font: { size: 14, color: '#334155' } },
            tickfont: { size: 12, color: '#475569' },
            gridcolor: '#e2e8f0'
        },
        height: 400,
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f8fafc',
        font: { color: '#1e293b' },
        margin: { l: 60, r: 40, t: 60, b: 60 }
    };
    
    Plotly.newPlot('batch-chart', data, layout, {responsive: true});
}

// Load Visualizations
async function loadVisualizations() {
    try {
        // Feature importance
        const importanceResponse = await fetch('/feature_importance');
        const importanceData = await importanceResponse.json();
        
        if (importanceData.success) {
            const data = [{
                type: 'bar',
                x: importanceData.importance,
                y: importanceData.features.map(f => f.replace('koi_', '').replace('_', ' ')),
                orientation: 'h',
                marker: { color: '#667eea' }
            }];
            
            const layout = {
                title: {
                    text: 'Feature Importance for Exoplanet Detection',
                    font: { size: 20, color: '#1e293b', weight: 'bold' }
                },
                xaxis: { 
                    title: { text: 'Importance Score', font: { size: 14, color: '#334155' } },
                    tickfont: { size: 12, color: '#475569' },
                    gridcolor: '#e2e8f0'
                },
                yaxis: { 
                    title: '',
                    tickfont: { size: 12, color: '#475569' }
                },
                height: 500,
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#f8fafc',
                font: { color: '#1e293b' },
                margin: { l: 150, r: 40, t: 60, b: 60 }
            };
            
            Plotly.newPlot('importance-chart', data, layout, {responsive: true});
        }
        
        // Model performance
        const perfResponse = await fetch('/model_performance');
        const perfData = await perfResponse.json();
        
        if (perfData.success) {
            const models = ['DNN', 'CNN', 'LSTM', 'Ensemble'];
            const accuracies = [
                perfData.performance.dnn.accuracy * 100,
                perfData.performance.cnn.accuracy * 100,
                perfData.performance.lstm.accuracy * 100,
                perfData.performance.ensemble.accuracy * 100
            ];
            const aucScores = [
                perfData.performance.dnn.roc_auc,
                perfData.performance.cnn.roc_auc,
                perfData.performance.lstm.roc_auc,
                perfData.performance.ensemble.roc_auc
            ];
            
            const trace1 = {
                x: models,
                y: accuracies,
                name: 'Accuracy (%)',
                type: 'bar',
                marker: { color: '#667eea' }
            };
            
            const trace2 = {
                x: models,
                y: aucScores.map(s => s * 100),
                name: 'ROC-AUC (scaled)',
                type: 'bar',
                marker: { color: '#34a853' }
            };
            
            const data = [trace1, trace2];
            
            const layout = {
                title: {
                    text: 'Model Performance Comparison',
                    font: { size: 20, color: '#1e293b', weight: 'bold' }
                },
                barmode: 'group',
                xaxis: { 
                    title: { text: 'Model', font: { size: 14, color: '#334155' } },
                    tickfont: { size: 12, color: '#475569' },
                    gridcolor: '#e2e8f0'
                },
                yaxis: { 
                    title: { text: 'Score', font: { size: 14, color: '#334155' } },
                    tickfont: { size: 12, color: '#475569' },
                    gridcolor: '#e2e8f0'
                },
                height: 500,
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#f8fafc',
                font: { color: '#1e293b' },
                legend: {
                    font: { size: 12, color: '#1e293b' },
                    bgcolor: '#ffffff',
                    bordercolor: '#e2e8f0',
                    borderwidth: 1
                },
                margin: { l: 60, r: 40, t: 60, b: 60 }
            };
            
            Plotly.newPlot('performance-chart', data, layout, {responsive: true});
        }
    } catch (error) {
        console.error('Error loading visualizations:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('NASA Exoplanet Detection System initialized');
    
    // Initialize all slider displays with real astronomical values
    const sliders = document.querySelectorAll('.input-group input[type="range"]');
    sliders.forEach(slider => {
        updateValue(slider.name);
    });
});
