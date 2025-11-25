from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from datetime import datetime

# Imports para análise
from scipy.fftpack import fft2, fftshift

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== MÉTODO 1: Luminância + Gradientes + PCA ====================
def method_luminance_gradient_pca(image_path):
    """
    Análise de luminância, gradientes e covariância via PCA
    Baseado no método do post original
    """
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Converter para luminância
        r, g, b = img_rgb[:,:,0]/255, img_rgb[:,:,1]/255, img_rgb[:,:,2]/255
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        
        # Calcular gradientes
        gx = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        
        # Flatten em matriz
        h, w = luminance.shape
        M = np.column_stack([gx.flatten(), gy.flatten()])
        
        # Covariância
        C = (1/len(M)) * (M.T @ M)
        
        # PCA - eigenvalues indicam estrutura
        eigenvalues = np.linalg.eigvalsh(C)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Score: razão entre eigenvalues
        # Imagens reais têm gradientes mais coerentes (eigenvalues mais balanceados)
        # IA gera estruturas instáveis (eigenvalues muito desiguais)
        ratio = eigenvalues[1] / (eigenvalues[0] + 1e-8)
        
        # Normalizar: ratio típico para imagens reais está entre 0.3-0.7
        # Para IA, tende a ser < 0.2 ou muito variável
        if ratio > 0.25:
            score = min(ratio * 1.5, 1.0)  # Escalar para favorecer real
        else:
            score = ratio * 2  # Penalizar muito se ratio baixo
        
        return {
            'method': 'Luminance Gradient PCA',
            'score': float(score),
            'confidence': 'Real' if score > 0.5 else 'AI Generated',
            'eigenvalue_ratio': float(ratio)
        }
    except Exception as e:
        return {'method': 'Luminance Gradient PCA', 'error': str(e), 'score': 0.5}

# ==================== MÉTODO 2: Análise Espectral (FFT) ====================
def method_spectral_analysis(image_path):
    """
    Análise de domínio de frequência via FFT
    Mede a "naturalidade" da distribuição espectral
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Reduzir para 512x512 para consistência (se maior)
        if max(img.shape) > 512:
            scale = 512 / max(img.shape)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # FFT 2D
        f_transform = fft2(img)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Criar perfil radial de energia (média em cada distância do centro)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)
        max_radius = int(np.sqrt(center_h**2 + center_w**2))
        
        # Calcular energia média por raio
        radial_profile = []
        for r in range(min(max_radius, 200)):  # Até raio 200
            mask = (dist_from_center == r)
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(magnitude_spectrum[mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # Imagens naturais têm decaimento exponencial/power-law
        # Calcular taxa de decaimento usando regressão linear no log
        if len(radial_profile) > 10:
            # Ignorar raio 0 (DC component muito alto)
            x_vals = np.arange(1, len(radial_profile))
            y_vals = np.log1p(radial_profile[1:])  # log para linearizar
            
            # Fit linear
            if len(x_vals) > 0 and np.std(y_vals) > 0:
                # Coeficiente de correlação
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                
                # Decaimento: quanto mais negativo, melhor (energia cai com raio)
                # Imagens reais: correlação negativa forte (-0.8 a -0.95)
                # IA: pode ter correlação menos negativa ou irregular
                
                # Normalizar correlação para score
                # -1.0 (perfeito decaimento) -> 1.0 (score alto)
                # 0 ou positivo (irregular) -> 0.0 (score baixo)
                if correlation < -0.5:
                    score = min(abs(correlation + 0.5) * 2, 1.0)
                else:
                    score = 0.3  # Penalizar se não tem decaimento claro
            else:
                score = 0.5
        else:
            score = 0.5
        
        return {
            'method': 'Spectral Analysis (FFT)',
            'score': float(score),
            'confidence': 'Real' if score > 0.5 else 'AI Generated',
            'spectral_decay_correlation': float(correlation) if 'correlation' in locals() else 0.0
        }
    except Exception as e:
        return {'method': 'Spectral Analysis', 'error': str(e), 'score': 0.5}

# ==================== MÉTODO 3: Análise de Textura (LBP) ====================
def method_texture_lbp(image_path):
    """
    Local Binary Pattern - detecta diferenças de textura
    Usa variância de LBP ao invés de entropia para melhor discriminação
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Reduzir imagem para acelerar (opcional)
        if img.shape[0] > 500 or img.shape[1] > 500:
            scale = 500 / max(img.shape)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Aplicar LBP manualmente (versão simplificada 3x3)
        h, w = img.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i, j]
                neighbors = [
                    img[i-1,j-1], img[i-1,j], img[i-1,j+1],
                    img[i,j+1], img[i+1,j+1], img[i+1,j],
                    img[i+1,j-1], img[i,j-1]
                ]
                lbp_val = 0
                for k, n in enumerate(neighbors):
                    lbp_val |= (1 << k) if n >= center else 0
                lbp[i-1, j-1] = lbp_val
        
        # Histogram do LBP
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        
        # Usar variância do histograma ao invés de entropia
        # Imagens reais têm variância moderada
        # IA pode ter variância muito baixa (padrões uniformes) ou muito alta (ruído artificial)
        hist_var = np.var(hist)
        
        # Também calcular o número de bins únicos usados
        unique_patterns = np.sum(hist > 0.001)  # bins com > 0.1% de pixels
        pattern_diversity = unique_patterns / 256
        
        # Score: combinar variância e diversidade
        # Valores típicos: var ~0.00005-0.0002, diversity ~0.4-0.7 para real
        var_score = np.clip(hist_var * 5000, 0, 1)
        diversity_score = pattern_diversity
        
        score = (var_score * 0.4 + diversity_score * 0.6)
        score = np.clip(score, 0, 1)
        
        return {
            'method': 'Texture LBP',
            'score': float(score),
            'confidence': 'Real' if score > 0.5 else 'AI Generated',
            'histogram_variance': float(hist_var),
            'pattern_diversity': float(pattern_diversity)
        }
    except Exception as e:
        return {'method': 'Texture LBP', 'error': str(e), 'score': 0.5}

# ==================== MÉTODO 4: Análise de Ruído ====================
def method_noise_analysis(image_path):
    """
    Detecta assinatura de ruído - imagens reais têm ruído natural/sensor
    IA tende a ser muito limpa ou com ruído artificial
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Converter para float64 ANTES do Laplacian
        img_float = img.astype(np.float64)
        
        # Estimar ruído usando Laplacian
        laplacian = cv2.Laplacian(img_float, cv2.CV_64F, ksize=3)
        noise_estimate = np.std(laplacian)
        
        # Calcular variância de alta frequência usando Sobel
        sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Noise consistency: calcular variância local do ruído
        # Dividir imagem em blocos e ver quão consistente é o ruído
        block_size = 32
        h, w = img.shape
        noise_vars = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = img_float[i:i+block_size, j:j+block_size]
                block_lap = cv2.Laplacian(block, cv2.CV_64F, ksize=3)
                noise_vars.append(np.var(block_lap))
        
        # Consistência do ruído: baixa variância entre blocos = ruído natural
        # Alta variância = ruído artificial ou muito limpo
        noise_consistency = 1.0 / (1.0 + np.std(noise_vars) / (np.mean(noise_vars) + 1e-8))
        
        # Presença de ruído: imagens reais têm ruído moderado
        # noise_estimate típico: 10-40 para real, <5 ou >50 para IA
        if 8 < noise_estimate < 60:
            noise_presence = min(noise_estimate / 30, 1.0)
        else:
            noise_presence = 0.3  # Penalizar extremos
        
        # Score final: combinar presença e consistência
        score = (noise_presence * 0.6 + noise_consistency * 0.4)
        score = np.clip(score, 0, 1)
        
        return {
            'method': 'Noise Analysis',
            'score': float(score),
            'confidence': 'Real' if score > 0.5 else 'AI Generated',
            'noise_estimate': float(noise_estimate),
            'noise_consistency': float(noise_consistency)
        }
    except Exception as e:
        return {'method': 'Noise Analysis', 'error': str(e), 'score': 0.5}

# ==================== ENSEMBLE: Combinar todos os métodos ====================
def analyze_image(image_path):
    """
    Executa todos os métodos e retorna score consolidado
    """
    results = []
    
    # Executar cada método
    method1 = method_luminance_gradient_pca(image_path)
    method2 = method_spectral_analysis(image_path)
    method3 = method_texture_lbp(image_path)
    method4 = method_noise_analysis(image_path)
    
    results = [method1, method2, method3, method4]
    
    # Pesos para cada método (ajustados baseado em confiabilidade)
    # Luminance+PCA e Noise são mais confiáveis
    weights = {
        'Luminance Gradient PCA': 0.35,  # Mais confiável
        'Spectral Analysis (FFT)': 0.20,  # Médio (ainda calibrando)
        'Texture LBP': 0.25,               # Bom para texturas
        'Noise Analysis': 0.20             # Bom para ruído natural
    }
    
    # Calcular score ponderado
    weighted_sum = 0
    total_weight = 0
    valid_methods = 0
    
    for result in results:
        if 'score' in result and 'error' not in result:
            method_name = result['method']
            weight = weights.get(method_name, 0.25)
            weighted_sum += result['score'] * weight
            total_weight += weight
            valid_methods += 1
    
    # Score final = média ponderada
    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.5
    
    # Calcular desvio padrão dos scores (detectar discrepância)
    valid_scores = [r['score'] for r in results if 'score' in r and 'error' not in r]
    score_std = np.std(valid_scores) if len(valid_scores) > 1 else 0
    
    # Ajustar confiança baseado em concordância entre métodos
    # Se os métodos discordam muito (std alto), reduzir confiança
    agreement_penalty = min(score_std * 2, 0.2)  # Max 20% penalty
    
    # Classificação final com thresholds ajustados
    if final_score > 0.55:
        classification = 'Real Image'
        # Se discordam muito, baixar confiança
        confidence_level = 'High' if score_std < 0.30 else 'Medium'
    elif final_score > 0.42:
        classification = 'Uncertain'
        confidence_level = 'Low'
    else:
        classification = 'AI Generated'
        confidence_level = 'High' if score_std < 0.30 else 'Medium'
    
    return {
        'final_score': float(final_score),
        'classification': classification,
        'confidence_level': confidence_level,
        'method_agreement': f"{(1 - score_std):.1%}" if score_std <= 1 else "Low",
        'score_deviation': float(score_std),
        'detailed_results': results,
        'timestamp': datetime.now().isoformat()
    }

# ==================== ROTAS FLASK ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def upload_and_analyze():
    """
    Endpoint para upload e análise de imagem
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP'}), 400
    
    try:
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_image(filepath)
        result['filename'] = filename
        
        os.remove(filepath)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '0.1.0-MVP'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)