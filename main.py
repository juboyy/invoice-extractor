import os
import tempfile
import base64
import datetime
import json
import statistics
from io import BytesIO
from functools import wraps
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import pypdfium2 as pdfium
import jwt
import jiwer
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv
import anthropic

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configuração da chave secreta
SECRET_KEY = os.environ.get('SECRET_KEY', 'sua_chave_secreta')
app.config['SECRET_KEY'] = SECRET_KEY

# Configurar limite máximo de conteúdo (exemplo: 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Base de dados fictícia de usuários
users = {
    "admin": generate_password_hash("password")
}

# Definições dos campos do JSON a serem preenchidos
FIELD_DEFINITIONS = (
    "Definições dos campos do JSON a serem preenchidos:\n\n"
    "1. numeroRps (String) - Número do RPS que gerou a nota fiscal de saída de serviço. Obrigatório: Sim\n"
    "2. numeroNota (String) - Número da nota fiscal de saída de serviço. Obrigatório: Sim\n"
    "3. dataEmissao (String) - Data de emissão da nota fiscal de saída (Formato: DD/MM/YYYY HH24:MI:SS). Obrigatório: Sim\n"
    "4. codigoSerie (String) - Código da série da nota fiscal de serviço. Obrigatório: Não\n"
    "5. descricaoSerie (String) - Descrição da série da nota fiscal de serviço. Obrigatório: Não\n"
    "6. codigoModelo (String) - Código do modelo da nota fiscal de serviço. Obrigatório: Sim\n"
    "7. descricaoModelo (String) - Descrição do modelo da nota fiscal de serviço. Obrigatório: Não\n"
    "8. cnpjCliente (String) - CNPJ do cliente da nota fiscal de serviço. Obrigatório: Não\n"
    "9. razaoCliente (String) - Razão social do cliente da nota fiscal de serviço. Obrigatório: Não\n"
    "10. codIbgeEstadoServico (String) - Código IBGE do estado da execução do serviço. Obrigatório: Sim\n"
    "11. codIbgeCidadeServico (String) - Código IBGE da cidade da execução do serviço. Obrigatório: Sim\n"
    "12. tipoTributacaoIss (String) - Tipo de Tributação do ISS (1 a 9). Obrigatório: Sim\n"
    "   Valores possíveis:\n"
    "     1. Tributado no Município\n"
    "     2. Tributado fora do Município\n"
    "     3. Tributado no Município Isento\n"
    "     4. Tributado fora do Município Isento\n"
    "     5. Tributado no Município Imune\n"
    "     6. Tributado fora do Município Imune\n"
    "     7. Tributado no Município Suspensa\n"
    "     8. Tributado fora do Município Suspensa\n"
    "     9. Exp Servicos\n"
    "13. valorNotaFiscal (BigDecimal) - Valor da nota fiscal de serviço. Obrigatório: Sim\n"
    "14. valorMulta (BigDecimal) - Valor da multa na nota fiscal de serviço. Obrigatório: Não\n"
    "15. valorDesconto (BigDecimal) - Valor do desconto na nota fiscal de serviço. Obrigatório: Não\n"
    "16. termoRecebimento (String) - Descrição do termo de recebimento da nota fiscal de serviço integrado com o Fusion (Oracle). Obrigatório: Não\n"
    "17. observacao (String) - Descrição da observação da nota fiscal de serviço. Obrigatório: Sim\n\n"
    "Servicos:\n"
    "1. codigoTipoServico (String) - Código do Tipo de Serviço na nota fiscal. Obrigatório: Sim\n"
    "2. descricaoTipoServico (String) - Descrição do Tipo de Serviço na nota fiscal. Obrigatório: Sim\n"
    "3. codigoServico (String) - Código do Serviço na nota fiscal. Obrigatório: Sim\n"
    "4. descricaoServico (String) - Descrição do Serviço na nota fiscal. Obrigatório: Sim\n"
    "5. quantidadeServico (BigDecimal) - Quantidade do serviço na nota fiscal. Obrigatório: Sim\n"
    "6. valorServico (BigDecimal) - Valor do serviço na nota fiscal. Obrigatório: Sim\n"
    "7. valorTotalServico (BigDecimal) - Valor total do serviço na nota fiscal. Obrigatório: Sim\n\n"
    "ImpostosRetido (Se houver):\n"
    "1. indicadorImposto (String) - Tipo de imposto (ex: COFINS, PIS/PASEP, ISS, INSS-PJ, INSS-PF, IRRF-PF, IRRF-PJ, CSLL). Obrigatório se houver informação no documento.\n"
    "2. codigoReceita (String) - Código da Receita. Obrigatório: Não\n"
    "3. indicadorRetencao (String) - Indica se o imposto possui Retenção. Obrigatório se houver informação no documento.\n"
    "4. vlrBaseImposto (BigDecimal) - Valor base do Imposto. Obrigatório se houver informação no documento.\n"
    "5. aliquotaImposto (BigDecimal) - Alíquota do Imposto. Obrigatório se houver informação no documento.\n"
    "6. vlrImposto (BigDecimal) - Valor do Imposto. Obrigatório se houver informação no documento.\n\n"
    "Titulos:\n"
    "1. numeroTitulo (String) - Informar o número do título. Obrigatório: Não\n"
    "2. dataVencimento (Data) - Data de vencimento do título (Formato: DD/MM/YYYY). Obrigatório: Não\n"
    "3. cnpjCpfCredorTitulo (String) - CNPJ/CPF do credor do título. Obrigatório: Não\n"
    "4. valorTitulo (BigDecimal) - Valor do título. Obrigatório: Não\n"
    "5. indicadorTipoTitulo (String) - Tipo do título ('P' - Título do credor principal, 'R' - Título de retenção). Obrigatório: Não\n"
)

@dataclass
class OCRResult:
    text: str
    confidences: List[float]

@dataclass
class ExtractionResult:
    text: str
    source: str  # 'anthropic' ou 'ocr'
    confidences: List[float] = None

def estimate_tokens(text):
    return len(text.split())

def convert_pdf_to_images(pdf_bytes):
    images_list = []
    tmp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_pdf_path = tmp_file.name

        pdf = pdfium.PdfDocument(tmp_pdf_path)
        for page_number in range(len(pdf)):
            page = pdf.get_page(page_number)
            pil_image = page.render(
                scale=300/72,  # DPI adequado para OCR
                rotation=0
            ).to_pil()
            
            pil_image = pil_image.convert('RGB')
            
            img_byte_arr = BytesIO()
            pil_image.save(
                img_byte_arr, 
                format='JPEG', 
                quality=95,
                optimize=True
            )
            
            images_list.append({
                'page_number': page_number + 1,
                'image_bytes': img_byte_arr.getvalue()
            })
            page.close()
        pdf.close()
        
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
            
    return images_list

def extract_text_with_anthropic(image_bytes):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return {"error": "A chave de API da Anthropic não foi fornecida."}

    try:
        client = anthropic.Client(api_key=api_key)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Por favor, extraia todo o texto presente na imagem. Retorne apenas o texto extraído, sem formatação ou comentários adicionais."
                        "Analise o texto a seguir e gere um JSON seguindo as definições de campos fornecidas. "
                        "Se alguma informação n��o estiver presente no texto, use null para o valor do campo. "
                        "Certifique-se de que o JSON gerado seja válido e siga estritamente as definições.\n\n"
                        f"Definições dos campos:\n{FIELD_DEFINITIONS}\n\n"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }]
        )
        
        return response.content[0].text if response.content else ""
        
    except Exception as e:
        return {"error": f"Erro ao chamar a API da Anthropic: {str(e)}"}

def extract_text_with_ocr(image_bytes) -> OCRResult:
    try:
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if image.size[0] < 1000:
            ratio = 1000 / image.size[0]
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        result = pytesseract.image_to_data(image, lang='por', output_type=pytesseract.Output.DICT)
        
        words = []
        confidences = []
        for i, word in enumerate(result['text']):
            if word.strip():
                words.append(word)
                confidences.append(float(result['conf'][i]))
        
        return OCRResult(
            text=' '.join(words).strip(),
            confidences=confidences
        )
        
    except Exception as e:
        print(f"Erro no OCR: {str(e)}")
        return OCRResult(text="", confidences=[])

def evaluate_extraction_metrics(extracted_result: ExtractionResult, ground_truth_text: str) -> Dict[str, Any]:
    """
    Avalia a qualidade da extração de texto com threshold de 95% de acurácia.
    """
    ACCURACY_THRESHOLD = 95.0

    def normalize_text_robust(text: str) -> str:
        """
        Normalização rigorosa de texto para comparação.
        Mantém apenas letras e números, removendo todo o resto.
        """
        import re
        from unicodedata import normalize
        
        if not text:
            return ""
            
        # Converter para minúsculas
        text = text.lower()
        
        # Remover acentos
        text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        
        # Substituir quebras de linha por espaços
        text = text.replace('\n', ' ')
        
        # Remover todos os caracteres especiais e pontuação, mantendo apenas letras e números
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Normalizar espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remover palavras muito curtas (artigos, preposições etc)
        words = [word for word in text.split() if len(word) > 2]
        
        # Remover números isolados
        words = [word for word in words if not word.isdigit()]
        
        return ' '.join(words)

    try:
        # Normalizar textos
        extracted_norm = normalize_text_robust(extracted_result.text)
        ground_truth_norm = normalize_text_robust(ground_truth_text)

        if not extracted_norm or not ground_truth_norm:
            return {
                "error": "Um dos textos está vazio após a normalização.",
                "atende_requisito": False,
                "textos_comparados": {
                    "texto_extraido": extracted_result.text,
                    "ground_truth": ground_truth_text
                }
            }

        # Análise de palavras usando conjuntos para comparação exata
        extracted_words = set(extracted_norm.split())
        ground_truth_words = set(ground_truth_norm.split())
        
        # Palavras em comum
        palavras_corretas = extracted_words & ground_truth_words
        palavras_ausentes = ground_truth_words - extracted_words
        palavras_extras = extracted_words - ground_truth_words
        
        # Cálculo de acurácia
        palavras_totais = len(ground_truth_words)
        word_accuracy = (len(palavras_corretas) / palavras_totais * 100) if palavras_totais > 0 else 0
        
        # Comparação de caracteres após normalização
        char_accuracy = 100 - jiwer.cer(ground_truth_norm, extracted_norm) * 100
        
        # Média ponderada
        accuracy_final = (word_accuracy * 0.7 + char_accuracy * 0.3)

        metrics = {
            "fonte_extracao": extracted_result.source,
            "atende_requisito": accuracy_final >= ACCURACY_THRESHOLD,
            "textos_comparados": {
                "texto_extraido": {
                    "original": extracted_result.text,
                    "normalizado": extracted_norm,
                    "palavras": sorted(list(extracted_words))
                },
                "ground_truth": {
                    "original": ground_truth_text,
                    "normalizado": ground_truth_norm,
                    "palavras": sorted(list(ground_truth_words))
                }
            },
            "metricas_principais": {
                "acuracia_final": round(accuracy_final, 2),
                "acuracia_palavras": round(word_accuracy, 2),
                "acuracia_caracteres": round(char_accuracy, 2),
                "threshold_requerido": ACCURACY_THRESHOLD
            },
            "metricas_detalhadas": {
                "palavras_corretas": len(palavras_corretas),
                "palavras_totais": palavras_totais,
                "palavras_ausentes": len(palavras_ausentes),
                "palavras_extras": len(palavras_extras),
                "palavras_ausentes_lista": sorted(list(palavras_ausentes)),
                "palavras_extras_lista": sorted(list(palavras_extras)),
                "palavras_corretas_lista": sorted(list(palavras_corretas))
            }
        }

        return metrics

    except Exception as e:
        return {
            "error": f"Erro na avaliação: {str(e)}",
            "atende_requisito": False,
            "textos_comparados": {
                "texto_extraido": extracted_result.text,
                "ground_truth": ground_truth_text
            }
        }

def generate_json_from_text(text):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return {"error": "A chave de API da Anthropic não foi fornecida."}

    try:
        client = anthropic.Client(api_key=api_key)
        
        prompt = (
            "Analise o texto a seguir e gere um JSON seguindo as definições de campos fornecidas. "
            "Se alguma informação não estiver presente no texto, use null para o valor do campo. "
            "Certifique-se de que o JSON gerado seja válido e siga estritamente as definições.\n\n"
            f"Definições dos campos:\n{FIELD_DEFINITIONS}\n\n"
            f"Texto para análise:\n{text}"
        )
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        response_text = response.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"error": "Não foi possível extrair um JSON válido da resposta"}
            
    except Exception as e:
        return {"error": f"Erro ao gerar JSON: {str(e)}"}

@app.route('/login', methods=['POST'])
def login():
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Credenciais incompletas'}), 401

    username = auth.get('username')
    password = auth.get('password')

    if username not in users:
        return jsonify({'message': 'Usuário não encontrado'}), 401

    if check_password_hash(users.get(username), password):
        token = jwt.encode({
            'user': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        
        return jsonify({'token': token})

    return jsonify({'message': 'Senha incorreta'}), 401

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'message': 'Token ausente'}), 403
            
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 403
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido'}), 403
            
        return f(*args, **kwargs)
    return decorated

def process_with_anthropic(file_bytes, file_type='image'):
    """
    Processa arquivo com Anthropic Claude, suportando PDF e imagens.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Erro: ANTHROPIC_API_KEY não encontrada")
        return {"error": "A chave de API da Anthropic não foi fornecida."}

    try:
        # Configurar cliente com header beta para PDFs
        client = anthropic.Anthropic(
            api_key=api_key,
            default_headers={
                "anthropic-beta": "pdfs-2024-09-25"
            }
        )
        
        file_base64 = base64.b64encode(file_bytes).decode('utf-8')
        MODEL_NAME = "claude-3-5-sonnet-20241022"
        
        # Prompt unificado
        prompt = (
            "Analise o documento e forneça duas coisas:\n"
            "1. O texto bruto extraído (raw_text)\n"
            "2. Um JSON estruturado seguindo as definições abaixo\n\n"
            f"Definições dos campos:\n{FIELD_DEFINITIONS}\n\n"
            "Formato da resposta:\n"
            "RAW_TEXT:\n"
            "<texto extraído>\n\n"
            "JSON:\n"
            "<json estruturado>"
        )

        if file_type == 'pdf':
            print("Processando PDF com API do Claude")
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": file_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        else:
            print("Processando imagem")
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": file_base64
                        }
                    }
                ]
            }]

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            temperature=0,
            messages=messages
        )
        
        if not response or not response.content:
            print("Erro: Resposta vazia da API")
            return {"error": "Resposta vazia da API Anthropic"}
            
        response_text = response.content[0].text
        print(f"Resposta recebida (tamanho: {len(response_text)})")
        
        # Extrair texto bruto e JSON da resposta
        raw_text_start = response_text.find('RAW_TEXT:')
        json_start = response_text.find('JSON:')
        
        if raw_text_start == -1 or json_start == -1:
            print("Erro: Formato de resposta inválido")
            return {"error": "Formato de resposta inválido"}
            
        raw_text_start += 9  # Comprimento de 'RAW_TEXT:'
        raw_text = response_text[raw_text_start:json_start].strip()
        json_str = response_text[json_start + 5:].strip()
        
        # Extrair JSON válido
        json_start = json_str.find('{')
        json_end = json_str.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            try:
                json_data = json.loads(json_str[json_start:json_end])
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {e}")
                json_data = {"error": "JSON inválido"}
        else:
            print("Erro: Não foi possível encontrar JSON válido")
            json_data = {"error": "Não foi possível extrair um JSON válido"}

        return {
            "raw_text": raw_text,
            "json_data": json_data,
            "source": "anthropic"
        }
        
    except Exception as e:
        print(f"Erro ao processar com Anthropic: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Erro ao processar com Anthropic: {str(e)}"}

@app.route('/process_document', methods=['POST'])
@token_required
def process_document():
    if 'file' not in request.files:
        return jsonify({'message': 'Um arquivo PDF ou imagem é necessário'}), 400

    file = request.files['file']
    ground_truth_file = request.files.get('txt')
    
    if not file.filename or '.' not in file.filename:
        return jsonify({'message': 'Arquivo inválido'}), 400

    try:
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        if file_ext not in ['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff']:
            return jsonify({'message': 'Formato de arquivo não suportado'}), 400

        file_bytes = file.read()
        print(f"Processando arquivo: {file.filename} ({file_ext})")
        
        # Processar ground truth se fornecido
        ground_truth_text = None
        if ground_truth_file:
            try:
                ground_truth_text = ground_truth_file.read().decode('utf-8')
            except UnicodeDecodeError:
                ground_truth_file.seek(0)
                ground_truth_text = ground_truth_file.read().decode('latin1')

        # Processar arquivo
        result = process_with_anthropic(file_bytes, file_type=file_ext)
        
        if 'error' in result:
            print(f"Erro no processamento principal: {result['error']}")
            if file_ext != 'pdf':  # Tentar OCR apenas para imagens
                print("Tentando fallback para OCR")
                ocr_result = extract_text_with_ocr(file_bytes)
                result = {
                    "raw_text": ocr_result.text,
                    "json_data": None,
                    "source": "ocr",
                    "confidences": ocr_result.confidences
                }
            else:
                return jsonify({'message': result['error']}), 500

        response_data = {
            "extracted_text": result.get('raw_text', ''),
            "generated_json": result.get('json_data'),
            "extraction_sources": [result.get('source', 'unknown')]
        }

        # Adicionar métricas apropriadas
        if ground_truth_text:
            # Usar avaliação com ground truth
            metrics = evaluate_extraction_metrics(
                ExtractionResult(
                    text=result['raw_text'],
                    source=result['source']
                ),
                ground_truth_text
            )
        else:
            # Usar nova avaliação de qualidade
            metrics = evaluate_extraction_quality(result)
            
        response_data["evaluation_metrics"] = [metrics]

        return jsonify(response_data)

    except Exception as e:
        print(f"Erro detalhado: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Erro ao processar documento: {str(e)}'}), 500

def evaluate_extraction_quality(extracted_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Avalia a qualidade da extração sem ground truth, retornando uma pontuação de 0 a 100.
    """
    try:
        score = 100  # Começa com pontuação máxima
        deductions = []  # Lista para armazenar as deduções e seus motivos
        
        raw_text = extracted_result.get('raw_text', '')
        json_data = extracted_result.get('json_data', {})
        
        if not raw_text or not json_data:
            return {
                "acuracia": 0,
                "motivo": "Texto extraído ou JSON ausente"
            }

        # 1. Verificar campos obrigatórios (peso: 40%)
        campos_obrigatorios = [
            'numeroRps', 'numeroNota', 'dataEmissao', 'codigoModelo',
            'codIbgeEstadoServico', 'codIbgeCidadeServico', 'tipoTributacaoIss',
            'valorNotaFiscal', 'observacao'
        ]
        
        campos_ausentes = [campo for campo in campos_obrigatorios if not json_data.get(campo)]
        if campos_ausentes:
            deducao = (len(campos_ausentes) / len(campos_obrigatorios)) * 40
            deductions.append({
                "tipo": "campos_obrigatorios",
                "deducao": deducao,
                "campos_ausentes": campos_ausentes
            })
            score -= deducao

        # 2. Validar formato dos campos (peso: 20%)
        validacoes_formato = {
            'dataEmissao': lambda x: len(x.split('/')) == 3 if x else False,
            'valorNotaFiscal': lambda x: isinstance(x, (int, float)) if x else False,
            'tipoTributacaoIss': lambda x: str(x) in ['1','2','3','4','5','6','7','8','9'] if x else False
        }
        
        erros_formato = []
        for campo, validacao in validacoes_formato.items():
            if campo in json_data and not validacao(json_data[campo]):
                erros_formato.append(campo)
        
        if erros_formato:
            deducao = (len(erros_formato) / len(validacoes_formato)) * 20
            deductions.append({
                "tipo": "formato_invalido",
                "deducao": deducao,
                "campos_invalidos": erros_formato
            })
            score -= deducao

        # 3. Verificar consistência dos dados (peso: 20%)
        inconsistencias = []
        
        # Verificar se valor total bate com soma dos serviços
        if 'servicos' in json_data and json_data.get('valorNotaFiscal'):
            valor_total_servicos = sum(
                servico.get('valorTotalServico', 0) 
                for servico in json_data['servicos']
            )
            if abs(valor_total_servicos - json_data['valorNotaFiscal']) > 0.01:
                inconsistencias.append("valor_total_inconsistente")

        if inconsistencias:
            deducao = (len(inconsistencias) / 3) * 20  # 3 é o número máximo de verificações
            deductions.append({
                "tipo": "inconsistencias",
                "deducao": deducao,
                "detalhes": inconsistencias
            })
            score -= deducao

        # 4. Avaliar qualidade do texto extraído (peso: 20%)
        problemas_texto = []
        
        # Verificar se tem conteúdo mínimo
        if len(raw_text.split()) < 50:  # Mínimo de 50 palavras
            problemas_texto.append("texto_muito_curto")
            
        # Verificar se tem caracteres estranhos ou mal formatados
        if len([c for c in raw_text if not c.isprintable()]) > len(raw_text) * 0.1:
            problemas_texto.append("caracteres_invalidos")

        if problemas_texto:
            deducao = (len(problemas_texto) / 2) * 20  # 2 é o número máximo de problemas
            deductions.append({
                "tipo": "qualidade_texto",
                "deducao": deducao,
                "problemas": problemas_texto
            })
            score -= deducao

        # Garantir que o score fique entre 0 e 100
        score = max(0, min(100, score))
        
        return {
            "acuracia": round(score, 2),
            "atende_requisito": score >= 95,
            "deducoes": deductions,
            "analise": {
                "campos_obrigatorios_ausentes": campos_ausentes if campos_ausentes else None,
                "campos_formato_invalido": erros_formato if erros_formato else None,
                "inconsistencias_encontradas": inconsistencias if inconsistencias else None,
                "problemas_texto": problemas_texto if problemas_texto else None
            },
            "recomendacao": (
                "Extração atende aos requisitos de qualidade"
                if score >= 95 else
                "Necessária revisão manual ou nova extração"
            )
        }

    except Exception as e:
        return {
            "acuracia": 0,
            "erro": f"Erro na avaliação: {str(e)}",
            "atende_requisito": False
        }

if __name__ == '__main__':
    app.run(debug=True)


