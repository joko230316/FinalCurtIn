#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
양자 머신러닝 기반 BTC 트레이딩 봇 - Ubuntu 버전
15분봉 & 4시간봉 멀티타임프레임 분석 + 레버리지 설정
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import ccxt
import warnings
import traceback
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import random
import string
import sys
import logging
from collections import deque
import platform
import hashlib

# === 리눅스 호환성 설정 ===
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

# 디렉토리 생성
BASE_DIR = Path(os.path.expanduser("~")) / "quantum_trading"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for directory in [BASE_DIR, DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 파일 경로 정의
MARKET_DATA_FILE_15M = DATA_DIR / "market_data_15m.csv"
MARKET_DATA_FILE_4H = DATA_DIR / "market_data_4h.csv"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.pkl"
MODEL_FILE_15M = MODEL_DIR / "quantum_model_15m.pkl"
MODEL_FILE_4H = MODEL_DIR / "quantum_model_4h.pkl"
PERFORMANCE_LOG_FILE = DATA_DIR / "performance_log.csv"
TRAILING_STOP_FILE = DATA_DIR / "trailing_stops.pkl"
PATTERN_LEARNING_FILE = DATA_DIR / "pattern_learning.pkl"


# 리눅스 호환 로깅 포맷
class LinuxCompatibleFormatter(logging.Formatter):
    def format(self, record):
        # 리눅스에서는 이모지 제거
        if IS_LINUX:
            record.msg = self._replace_emojis(record.msg)
        return super().format(record)

    def _replace_emojis(self, text):
        if not isinstance(text, str):
            return text

        replacements = {
            '🚀': '[LAUNCH]', '📈': '[UP]', '✅': '[OK]', '❌': '[ERROR]',
            '⚠️': '[WARN]', '🔮': '[QUANTUM]', '🎯': '[TARGET]', '📊': '[CHART]',
            '📡': '[RADAR]', '⏰': '[CLOCK]', '🔄': '[SYNC]', '🔒': '[LOCK]',
            '📦': '[PACKAGE]', '🆕': '[NEW]', '⏸️': '[PAUSE]', '🎲': '[DICE]',
            '💾': '[SAVE]', '📁': '[FOLDER]', '🛑': '[STOP]', '🔥': '[FIRE]',
            '🤖': '[ROBOT]', '🖥️': '[PC]', '🔧': '[TOOL]', '💡': '[IDEA]',
            '📚': '[BOOKS]', '⚡': '[ZAP]', '🎨': '[ART]', '🔍': '[SEARCH]',
            '💰': '[MONEY]', '📉': '[DOWN]', '🎪': '[CIRCUS]', '🏆': '[TROPHY]',
            '🔔': '[BELL]', '📝': '[NOTE]', '📌': '[PIN]', '📍': '[LOCATION]',
            '🕒': '[TIME]', '🌟': '[STAR]', '⭐': '[STAR]', '🌙': '[MOON]',
            '☀️': '[SUN]', '🎉': '[PARTY]', '🔑': '[KEY]', '🚪': '[DOOR]',
            '🎢': '[ROLLER]', '🔄': '[TRAILING]', '📉': '[FALL]',
            '🧠': '[BRAIN]', '📚': '[LEARN]', '🎯': '[BONUS]', '📊': '[ANALYSIS]',
            '⚖️': '[LEVERAGE]'
        }

        for emoji, replacement in replacements.items():
            text = text.replace(emoji, replacement)
        return text


# 로깅 설정
LOG_FILE = LOG_DIR / "quantum_trading_bot.log"
log_formatter = LinuxCompatibleFormatter('%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

for handler in logging.getLogger().handlers:
    handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# === OKX 실거래 API 인증 ===
API_KEY = os.getenv("OKXYH_API_KEY")
API_SECRET = os.getenv("OKXYH_API_SECRET")
API_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

# API 키가 없는 경우 환경변수에서 로드 시도
if not API_KEY:
    API_KEY = os.getenv("OKX_API_KEY")
if not API_SECRET:
    API_SECRET = os.getenv("OKX_API_SECRET")
if not API_PASSPHRASE:
    API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")

# 환경변수에도 없는 경우 샌드박스 모드
if not API_KEY or not API_SECRET or not API_PASSPHRASE:
    logger.warning("[WARN] 환경변수에서 API 인증 정보를 찾을 수 없습니다.")
    logger.info("[INFO] .env 파일을 생성하거나 환경변수를 설정해주세요.")
    USE_SANDBOX = True
else:
    USE_SANDBOX = False

# OKX 거래소 설정
exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSPHRASE,
    'sandbox': USE_SANDBOX,
    'enableRateLimit': True,
}

exchange = ccxt.okx(exchange_config)

# === 거래 설정 ===
SYMBOL = "BTC/USDT:USDT"  # OKX 선물 심볼
TIMEFRAME_15M = "15m"  # 15분봉
TIMEFRAME_4H = "4h"  # 4시간봉
CANDLE_LIMIT = 200
CONTRACT_AMOUNT = 0.1
INTERVAL_NORMAL = 900  # 15분 (15분봉 주기에 맞춤)
INTERVAL_ACTIVE = 300  # 5분 (활성 거래 시)
INTERVAL_WAITING = 600  # 10분 (대기 모드)
TRADING_MODE = "cross"

# 레버리지 설정 (기본값 100x, 명령줄에서 조정 가능)
LEVERAGE = 100

# 손익 관리 설정 (레버리지 반영)
TAKE_PROFIT_PERCENT = 5000.0 / LEVERAGE  # 레버리지 반영
STOP_LOSS_PERCENT = -5000.0 / LEVERAGE  # 레버리지 반영
TRAILING_STOP_PERCENT = 5.0  # 트레일링 스탑 5%
EMERGENCY_LIQUIDATION_THRESHOLD = -10.0
QUANTUM_FEATURE_DIMENSION = 5  # 특징 개수 5개로 수정 (패턴 특징 포함)
SIGNAL_MATCH_THRESHOLD = 0.7  # 70% 이상 일치 시 매수

# === 강화학습 및 패턴 학습 설정 ===
REINFORCEMENT_WEIGHT = 1.5  # 수익성 패턴 가중치
MIN_PATTERN_OCCURRENCE = 3  # 최소 패턴 발생 횟수
PATTERN_SIMILARITY_THRESHOLD = 0.8  # 패턴 유사도 임계값

# 성과 데이터 구조
PERFORMANCE_DATA = {
    'total_trades': 0,
    'total_pnl': 0.0,
    'winning_trades': 0,
    'initial_balance': 0.0,
    'recent_trades': deque(maxlen=100),
    'signal_accuracy_history': deque(maxlen=50),
    'similar_trade_patterns': {},
    'market_regime_performance': {},
    'time_based_performance': {}
}

# 트레일링 스탑 데이터 구조
TRAILING_STOPS = {}

# 패턴 학습 데이터 구조
PATTERN_LEARNING_DATA = {
    'profitable_patterns': {},
    'unprofitable_patterns': {},
    'pattern_weights': {},
    'learning_history': [],
    'last_retrain': None
}
#-------
def load_okx_trade_history():
    """OKX 거래 내역을 로드하여 학습 데이터에 통합"""
    try:
        history_file = DATA_DIR / "okx_trade_history.pkl"
        if history_file.exists():
            with open(history_file, 'rb') as f:
                okx_trades = pickle.load(f)

            logger.info(f"[LEARN] OKX 거래 내역 로드: {len(okx_trades)}개 거래")

            # 기존 학습 데이터와 통합
            for trade in okx_trades:
                # 패턴 학습 데이터 업데이트
                update_pattern_learning(trade)

            logger.info("[LEARN] OKX 거래 내역 학습 데이터 통합 완료")
            return True
        else:
            logger.info("[LEARN] OKX 거래 내역 파일이 없습니다")
            return False
    except Exception as e:
        logger.error(f"[ERROR] OKX 거래 내역 로드 실패: {e}")
        return False


# 메인 함수 수정 (main() 함수 내에 추가)
def main():
    # ... 기존 코드 ...

    # OKX 거래 내역 로드 및 학습
    load_okx_trade_history()

    # ... 나머지 코드 ...
#-------
# === 패턴 학습 관리 함수 ===
def load_pattern_learning():
    """패턴 학습 데이터 로드"""
    global PATTERN_LEARNING_DATA
    try:
        if PATTERN_LEARNING_FILE.exists():
            with open(PATTERN_LEARNING_FILE, 'rb') as f:
                PATTERN_LEARNING_DATA = pickle.load(f)
            logger.info("[LEARN] 패턴 학습 데이터 로드 완료")
            logger.info(f"[LEARN] 수익성 패턴: {len(PATTERN_LEARNING_DATA['profitable_patterns'])}개")
            logger.info(f"[LEARN] 비수익성 패턴: {len(PATTERN_LEARNING_DATA['unprofitable_patterns'])}개")
            logger.info(f"[LEARN] 패턴 가중치: {len(PATTERN_LEARNING_DATA['pattern_weights'])}개")
    except Exception as e:
        logger.error(f"[ERROR] 패턴 학습 데이터 로드 실패: {e}")
        PATTERN_LEARNING_DATA = {
            'profitable_patterns': {},
            'unprofitable_patterns': {},
            'pattern_weights': {},
            'learning_history': [],
            'last_retrain': None
        }


def save_pattern_learning():
    """패턴 학습 데이터 저장"""
    try:
        with open(PATTERN_LEARNING_FILE, 'wb') as f:
            pickle.dump(PATTERN_LEARNING_DATA, f)
        logger.debug("[LEARN] 패턴 학습 데이터 저장 완료")
    except Exception as e:
        logger.error(f"[ERROR] 패턴 학습 데이터 저장 실패: {e}")


def generate_pattern_hash(pattern):
    """패턴 해시 생성"""
    pattern_str = ''.join(map(str, pattern))
    return hashlib.md5(pattern_str.encode()).hexdigest()[:12]


def calculate_pattern_similarity(pattern1, pattern2):
    """두 패턴 간 유사도 계산"""
    if len(pattern1) != len(pattern2):
        return 0

    matches = sum(1 for p1, p2 in zip(pattern1, pattern2) if p1 == p2)
    return matches / len(pattern1)


def find_similar_pattern(current_pattern, pattern_dict, threshold=PATTERN_SIMILARITY_THRESHOLD):
    """유사한 패턴 찾기"""
    if not current_pattern:
        return None, 0

    best_similarity = 0
    best_pattern = None
    best_pattern_id = None

    for pattern_id, pattern_data in pattern_dict.items():
        similarity = calculate_pattern_similarity(current_pattern, pattern_data['pattern'])
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_pattern = pattern_data
            best_pattern_id = pattern_id

    return best_pattern_id, best_similarity


def update_pattern_learning(trade_record):
    """패턴 학습 업데이트"""
    try:
        pattern = trade_record.get('trade_pattern')
        pnl_usdt = trade_record.get('pnl_usdt', 0)
        pnl_percent = trade_record.get('pnl_percent', 0)

        if not pattern:
            return

        pattern_hash = generate_pattern_hash(pattern)

        # 패턴 데이터 준비
        pattern_data = {
            'pattern': pattern,
            'pnl_usdt': pnl_usdt,
            'pnl_percent': pnl_percent,
            'count': 1,
            'total_pnl': pnl_usdt,
            'avg_pnl': pnl_usdt,
            'last_seen': datetime.now().isoformat(),
            'market_regime': trade_record.get('market_regime', 'UNKNOWN'),
            'hour_of_day': trade_record.get('hour_of_day', 0)
        }

        # 수익성 기준 (0.1% 이상 수익)
        is_profitable = pnl_percent > 0.1

        if is_profitable:
            # 수익성 패턴 업데이트
            if pattern_hash in PATTERN_LEARNING_DATA['profitable_patterns']:
                existing = PATTERN_LEARNING_DATA['profitable_patterns'][pattern_hash]
                existing['count'] += 1
                existing['total_pnl'] += pnl_usdt
                existing['avg_pnl'] = existing['total_pnl'] / existing['count']
                existing['last_seen'] = datetime.now().isoformat()
                logger.info(
                    f"[LEARN] 기존 수익성 패턴 업데이트: {pattern_hash} (횟수: {existing['count']}, 평균 PnL: {existing['avg_pnl']:.3f} USDT)")
            else:
                PATTERN_LEARNING_DATA['profitable_patterns'][pattern_hash] = pattern_data
                logger.info(f"[LEARN] 새로운 수익성 패턴 등록: {pattern_hash} (PnL: {pnl_usdt:.3f} USDT)")

            # 패턴 가중치 업데이트 (수익성 패턴은 가중치 증가)
            current_weight = PATTERN_LEARNING_DATA['pattern_weights'].get(pattern_hash, 1.0)
            new_weight = min(current_weight * REINFORCEMENT_WEIGHT, 5.0)  # 최대 5배까지
            PATTERN_LEARNING_DATA['pattern_weights'][pattern_hash] = new_weight

            logger.info(f"[LEARN] 수익성 패턴 가중치 증가: {pattern_hash} ({current_weight:.2f}x → {new_weight:.2f}x)")

        else:
            # 비수익성 패턴 업데이트
            if pattern_hash in PATTERN_LEARNING_DATA['unprofitable_patterns']:
                existing = PATTERN_LEARNING_DATA['unprofitable_patterns'][pattern_hash]
                existing['count'] += 1
                existing['total_pnl'] += pnl_usdt
                existing['avg_pnl'] = existing['total_pnl'] / existing['count']
                existing['last_seen'] = datetime.now().isoformat()
                logger.info(
                    f"[LEARN] 기존 비수익성 패턴 업데이트: {pattern_hash} (횟수: {existing['count']}, 평균 PnL: {existing['avg_pnl']:.3f} USDT)")
            else:
                PATTERN_LEARNING_DATA['unprofitable_patterns'][pattern_hash] = pattern_data
                logger.info(f"[LEARN] 새로운 비수익성 패턴 등록: {pattern_hash} (PnL: {pnl_usdt:.3f} USDT)")

            # 패턴 가중치 업데이트 (비수익성 패턴은 가중치 감소)
            current_weight = PATTERN_LEARNING_DATA['pattern_weights'].get(pattern_hash, 1.0)
            new_weight = max(current_weight / REINFORCEMENT_WEIGHT, 0.2)  # 최소 0.2배까지
            PATTERN_LEARNING_DATA['pattern_weights'][pattern_hash] = new_weight

            logger.info(f"[LEARN] 비수익성 패턴 가중치 감소: {pattern_hash} ({current_weight:.2f}x → {new_weight:.2f}x)")

        # 학습 기록 저장
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'pattern_hash': pattern_hash,
            'pattern': pattern,
            'pnl_usdt': pnl_usdt,
            'pnl_percent': pnl_percent,
            'is_profitable': is_profitable,
            'new_weight': PATTERN_LEARNING_DATA['pattern_weights'].get(pattern_hash, 1.0)
        }
        PATTERN_LEARNING_DATA['learning_history'].append(learning_record)

        # 최근 1000개 기록만 유지
        if len(PATTERN_LEARNING_DATA['learning_history']) > 1000:
            PATTERN_LEARNING_DATA['learning_history'] = PATTERN_LEARNING_DATA['learning_history'][-1000:]

        save_pattern_learning()

    except Exception as e:
        logger.error(f"[ERROR] 패턴 학습 업데이트 실패: {e}")


def get_pattern_bonus(current_pattern, market_regime, hour_of_day):
    """패턴 보너스 점수 계산"""
    if not current_pattern:
        return 0

    bonus_score = 0
    pattern_hash = generate_pattern_hash(current_pattern)

    # 정확히 일치하는 패턴 검색
    if pattern_hash in PATTERN_LEARNING_DATA['profitable_patterns']:
        pattern_data = PATTERN_LEARNING_DATA['profitable_patterns'][pattern_hash]
        weight = PATTERN_LEARNING_DATA['pattern_weights'].get(pattern_hash, 1.0)

        if pattern_data['count'] >= MIN_PATTERN_OCCURRENCE:
            # 기본 보너스 + 가중치 적용
            base_bonus = min(pattern_data['avg_pnl'] * 10, 0.5)  # 최대 0.5점
            regime_bonus = 0.1 if pattern_data['market_regime'] == market_regime else 0
            time_bonus = 0.05 if pattern_data['hour_of_day'] == hour_of_day else 0

            bonus_score = (base_bonus + regime_bonus + time_bonus) * weight
            logger.info(
                f"[BONUS] 정확한 패턴 매칭: {pattern_hash} (보너스: {bonus_score:.3f}, 가중치: {weight:.2f}x, 횟수: {pattern_data['count']})")

    # 유사 패턴 검색 (정확히 일치하지 않는 경우)
    else:
        similar_pattern_id, similarity = find_similar_pattern(
            current_pattern,
            PATTERN_LEARNING_DATA['profitable_patterns']
        )

        if similar_pattern_id and similarity >= 0.7:  # 70% 이상 유사
            pattern_data = PATTERN_LEARNING_DATA['profitable_patterns'][similar_pattern_id]
            weight = PATTERN_LEARNING_DATA['pattern_weights'].get(similar_pattern_id, 1.0)

            if pattern_data['count'] >= MIN_PATTERN_OCCURRENCE:
                # 유사도에 따른 보너스
                base_bonus = min(pattern_data['avg_pnl'] * 10, 0.3) * similarity
                regime_bonus = 0.1 if pattern_data['market_regime'] == market_regime else 0
                time_bonus = 0.05 if pattern_data['hour_of_day'] == hour_of_day else 0

                bonus_score = (base_bonus + regime_bonus + time_bonus) * weight
                logger.info(
                    f"[BONUS] 유사 패턴 매칭: {similar_pattern_id} (유사도: {similarity:.1%}, 보너스: {bonus_score:.3f}, 횟수: {pattern_data['count']})")

    return min(bonus_score, 1.0)  # 최대 1.0점 제한


def retrain_models_with_patterns():
    """패턴 데이터로 모델 재학습"""
    try:
        logger.info("[LEARN] 패턴 학습 데이터로 모델 재학습 시작...")

        # 최근 재학습 시간 확인 (24시간마다 재학습)
        current_time = datetime.now()
        last_retrain = PATTERN_LEARNING_DATA.get('last_retrain')
        if last_retrain:
            last_retrain_time = datetime.fromisoformat(last_retrain)
            hours_since_retrain = (current_time - last_retrain_time).total_seconds() / 3600
            if hours_since_retrain < 24:
                logger.info(f"[LEARN] 최근 재학습 이후 {hours_since_retrain:.1f}시간 경과, 재학습 스킵")
                return

        # 데이터 수집
        df_15m = fetch_ohlcv(timeframe=TIMEFRAME_15M, limit=1000)
        df_4h = fetch_ohlcv(timeframe=TIMEFRAME_4H, limit=1000)

        if df_15m is None or df_4h is None:
            logger.warning("[LEARN] 재학습을 위한 데이터 부족")
            return

        # 15분봉 모델 재학습
        quantum_model_15m = QuantumTradingModel(timeframe="15m")
        quantum_model_15m.train(df_15m, force_retrain=True)

        # 4시간봉 모델 재학습
        quantum_model_4h = QuantumTradingModel(timeframe="4h")
        quantum_model_4h.train(df_4h, force_retrain=True)

        PATTERN_LEARNING_DATA['last_retrain'] = current_time.isoformat()
        save_pattern_learning()

        logger.info("[LEARN] 모델 재학습 완료")

    except Exception as e:
        logger.error(f"[ERROR] 모델 재학습 실패: {e}")


def analyze_pattern_performance():
    """패턴 성과 분석"""
    try:
        profitable_count = len(PATTERN_LEARNING_DATA['profitable_patterns'])
        unprofitable_count = len(PATTERN_LEARNING_DATA['unprofitable_patterns'])
        total_patterns = profitable_count + unprofitable_count

        if total_patterns == 0:
            logger.info("[ANALYSIS] 분석할 패턴 데이터가 없습니다.")
            return

        # 상위 수익성 패턴 찾기
        profitable_patterns = list(PATTERN_LEARNING_DATA['profitable_patterns'].values())
        profitable_patterns.sort(key=lambda x: x['avg_pnl'], reverse=True)

        logger.info("[ANALYSIS] === 패턴 성과 분석 리포트 ===")
        logger.info(f"[ANALYSIS] 총 패턴: {total_patterns}개")
        logger.info(f"[ANALYSIS] 수익성 패턴: {profitable_count}개 ({profitable_count / total_patterns:.1%})")
        logger.info(f"[ANALYSIS] 비수익성 패턴: {unprofitable_count}개 ({unprofitable_count / total_patterns:.1%})")

        if profitable_patterns:
            top_patterns = profitable_patterns[:5]  # 상위 5개 패턴
            logger.info("[ANALYSIS] --- 상위 수익성 패턴 ---")
            for i, pattern in enumerate(top_patterns, 1):
                pattern_hash = generate_pattern_hash(pattern['pattern'])
                weight = PATTERN_LEARNING_DATA['pattern_weights'].get(pattern_hash, 1.0)
                logger.info(
                    f"[ANALYSIS] Top {i}: 해시={pattern_hash}, 평균 PnL={pattern['avg_pnl']:.3f} USDT, 횟수={pattern['count']}회, 가중치={weight:.2f}x")

        # 패턴 가중치 분포 분석
        weights = list(PATTERN_LEARNING_DATA['pattern_weights'].values())
        if weights:
            avg_weight = np.mean(weights)
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            high_weight_count = len([w for w in weights if w > 2.0])  # 2.0x 이상 가중치

            logger.info("[ANALYSIS] --- 패턴 가중치 분석 ---")
            logger.info(f"[ANALYSIS] 평균 가중치: {avg_weight:.2f}x")
            logger.info(f"[ANALYSIS] 최대 가중치: {max_weight:.2f}x")
            logger.info(f"[ANALYSIS] 최소 가중치: {min_weight:.2f}x")
            logger.info(f"[ANALYSIS] 고가중치 패턴(2.0x↑): {high_weight_count}개")

        # 최근 학습 활동
        recent_learnings = PATTERN_LEARNING_DATA['learning_history'][-10:]  # 최근 10개 학습
        if recent_learnings:
            logger.info("[ANALYSIS] --- 최근 학습 활동 ---")
            for learning in recent_learnings[-5:]:  # 최근 5개만 표시
                status = "수익" if learning['is_profitable'] else "손실"
                logger.info(
                    f"[ANALYSIS] {learning['timestamp'][11:19]} - {learning['pattern_hash']} - {status} - 가중치: {learning['new_weight']:.2f}x")

        logger.info("[ANALYSIS] === 분석 완료 ===")

    except Exception as e:
        logger.error(f"[ERROR] 패턴 성과 분석 실패: {e}")


# === 트레일링 스탑 관리 함수 ===
def load_trailing_stops():
    """트레일링 스탑 데이터 로드"""
    global TRAILING_STOPS
    try:
        if TRAILING_STOP_FILE.exists():
            with open(TRAILING_STOP_FILE, 'rb') as f:
                TRAILING_STOPS = pickle.load(f)
            logger.info("[TRAILING] 트레일링 스탑 데이터 로드 완료")
    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 데이터 로드 실패: {e}")
        TRAILING_STOPS = {}


def save_trailing_stops():
    """트레일링 스탑 데이터 저장"""
    try:
        with open(TRAILING_STOP_FILE, 'wb') as f:
            pickle.dump(TRAILING_STOPS, f)
        logger.debug("[TRAILING] 트레일링 스탑 데이터 저장 완료")
    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 데이터 저장 실패: {e}")


def initialize_trailing_stop(position_id, entry_price, side, current_price):
    """트레일링 스탑 초기화"""
    try:
        if side == "long":
            trailing_stop_price = entry_price * (1 - TRAILING_STOP_PERCENT / 100)
            highest_price = current_price
        else:  # short
            trailing_stop_price = entry_price * (1 + TRAILING_STOP_PERCENT / 100)
            lowest_price = current_price

        TRAILING_STOPS[position_id] = {
            'position_id': position_id,
            'entry_price': entry_price,
            'side': side,
            'trailing_stop_price': trailing_stop_price,
            'highest_price': highest_price if side == "long" else None,
            'lowest_price': lowest_price if side == "short" else None,
            'activated': False,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        logger.info(f"[TRAILING] 트레일링 스탑 초기화: {position_id}")
        logger.info(f"[TRAILING] 진입가: {entry_price:.2f}, 트레일링 스탑가: {trailing_stop_price:.2f}")
        save_trailing_stops()

    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 초기화 실패: {e}")


def update_trailing_stop(position_id, current_price):
    """트레일링 스탑 업데이트"""
    try:
        if position_id not in TRAILING_STOPS:
            return False

        trailing_data = TRAILING_STOPS[position_id]
        side = trailing_data['side']

        if side == "long":
            # 최고가 업데이트
            if current_price > trailing_data['highest_price']:
                trailing_data['highest_price'] = current_price
                # 트레일링 스탑가 업데이트
                new_trailing_stop = current_price * (1 - TRAILING_STOP_PERCENT / 100)
                if new_trailing_stop > trailing_data['trailing_stop_price']:
                    trailing_data['trailing_stop_price'] = new_trailing_stop
                    trailing_data['activated'] = True
                    logger.info(f"[TRAILING] LONG 트레일링 스탑 업데이트: {new_trailing_stop:.2f}")

            # 청산 조건 체크
            if current_price <= trailing_data['trailing_stop_price']:
                logger.info(
                    f"[TRAILING] LONG 트레일링 스탑 청산 조건 충족: {current_price:.2f} <= {trailing_data['trailing_stop_price']:.2f}")
                return True

        else:  # short
            # 최저가 업데이트
            if current_price < trailing_data['lowest_price']:
                trailing_data['lowest_price'] = current_price
                # 트레일링 스탑가 업데이트
                new_trailing_stop = current_price * (1 + TRAILING_STOP_PERCENT / 100)
                if new_trailing_stop < trailing_data['trailing_stop_price']:
                    trailing_data['trailing_stop_price'] = new_trailing_stop
                    trailing_data['activated'] = True
                    logger.info(f"[TRAILING] SHORT 트레일링 스탑 업데이트: {new_trailing_stop:.2f}")

            # 청산 조건 체크
            if current_price >= trailing_data['trailing_stop_price']:
                logger.info(
                    f"[TRAILING] SHORT 트레일링 스탑 청산 조건 충족: {current_price:.2f} >= {trailing_data['trailing_stop_price']:.2f}")
                return True

        trailing_data['updated_at'] = datetime.now().isoformat()
        return False

    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 업데이트 실패: {e}")
        return False


def remove_trailing_stop(position_id):
    """트레일링 스탑 제거"""
    try:
        if position_id in TRAILING_STOPS:
            del TRAILING_STOPS[position_id]
            save_trailing_stops()
            logger.info(f"[TRAILING] 트레일링 스탑 제거: {position_id}")
    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 제거 실패: {e}")


def check_all_trailing_stops(current_price):
    """모든 트레일링 스탑 체크"""
    try:
        positions_to_close = []
        for position_id, trailing_data in TRAILING_STOPS.items():
            if update_trailing_stop(position_id, current_price):
                positions_to_close.append(position_id)
        return positions_to_close
    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 체크 실패: {e}")
        return []


# === 기술적 지표 함수 ===
def calculate_rsi(prices, period=14):
    """RSI 계산"""
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logger.error(f"RSI 계산 오류: {e}")
        return pd.Series([50] * len(prices), index=prices.index)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    try:
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    except Exception as e:
        logger.error(f"MACD 계산 오류: {e}")
        empty_series = pd.Series([0] * len(prices), index=prices.index)
        return empty_series, empty_series, empty_series


def calculate_atr(high, low, close, period=14):
    """ATR 계산"""
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    except Exception as e:
        logger.error(f"ATR 계산 오류: {e}")
        return pd.Series([0] * len(high), index=high.index)


def add_technical_indicators(df):
    """데이터프레임에 기술적 지표 추가"""
    df = df.copy()
    try:
        df['return'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['rsi'] = calculate_rsi(df['close'])

        macd, macd_signal, _ = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal

        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        df['volatility'] = df['return'].rolling(20).std()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
    except Exception as e:
        logger.error(f"기술적 지표 계산 중 오류: {e}")
    return df


# === 파일 관리 함수 ===
def save_market_data(df, timeframe):
    """시장 데이터 저장"""
    try:
        if df is not None and len(df) > 0:
            file_path = MARKET_DATA_FILE_15M if timeframe == "15m" else MARKET_DATA_FILE_4H
            if file_path.exists():
                existing_data = pd.read_csv(file_path)
                combined_data = pd.concat([existing_data, df], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['ts'], keep='last')
            else:
                combined_data = df

            if len(combined_data) > 10000:
                combined_data = combined_data.tail(10000)

            combined_data.to_csv(file_path, index=False)
            logger.info(f"[FOLDER] {timeframe} 시장 데이터 저장 완료: {len(combined_data)}개 캔들")
    except Exception as e:
        logger.error(f"시장 데이터 저장 실패: {e}")


def load_market_data(timeframe):
    """시장 데이터 로드"""
    try:
        file_path = MARKET_DATA_FILE_15M if timeframe == "15m" else MARKET_DATA_FILE_4H
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            logger.info(f"[FOLDER] {timeframe} 시장 데이터 로드 완료: {len(df)}개 캔들")
            return df
        return None
    except Exception as e:
        logger.error(f"시장 데이터 로드 실패: {e}")
        return None


def load_trade_history():
    """거래 기록 불러오기"""
    try:
        if TRADE_HISTORY_FILE.exists():
            with open(TRADE_HISTORY_FILE, 'rb') as f:
                data = pickle.load(f)
                if 'recent_trades' in data and isinstance(data['recent_trades'], list):
                    data['recent_trades'] = deque(data['recent_trades'], maxlen=100)
                if 'signal_accuracy_history' in data and isinstance(data['signal_accuracy_history'], list):
                    data['signal_accuracy_history'] = deque(data['signal_accuracy_history'], maxlen=50)
                logger.info(f"[CHART] 거래 기록 로드 완료: {data['total_trades']}회 거래")
                return data
    except Exception as e:
        logger.error(f"거래 기록 불러오기 실패: {e}")
    return PERFORMANCE_DATA.copy()


def save_trade_history():
    """거래 기록 저장"""
    try:
        save_data = PERFORMANCE_DATA.copy()
        save_data['recent_trades'] = list(save_data['recent_trades'])
        save_data['signal_accuracy_history'] = list(save_data['signal_accuracy_history'])

        with open(TRADE_HISTORY_FILE, 'wb') as f:
            pickle.dump(save_data, f)
        logger.debug("거래 기록 저장 완료")
    except Exception as e:
        logger.error(f"거래 기록 저장 실패: {e}")


def save_model(model, scaler, timeframe):
    """모델 저장"""
    try:
        model_data = {
            'model': model,
            'scaler': scaler,
            'saved_at': datetime.now().isoformat(),
            'timeframe': timeframe
        }
        model_file = MODEL_FILE_15M if timeframe == "15m" else MODEL_FILE_4H
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"[SAVE] {timeframe} 모델 저장 완료")
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")


def load_model(timeframe):
    """모델 로드"""
    try:
        model_file = MODEL_FILE_15M if timeframe == "15m" else MODEL_FILE_4H
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"[SAVE] {timeframe} 모델 로드 완료")
            return model_data['model'], model_data['scaler']
        return None, None
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None, None


def log_performance(metrics):
    """성과 지표 로깅"""
    try:
        with open(PERFORMANCE_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()},{metrics}\n")
    except Exception as e:
        logger.error(f"성과 로깅 실패: {e}")


# === 머신러닝 관련 클래스 ===
class SimpleScaler:
    """MinMaxScaler 대체 구현"""

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        if len(X) == 0:
            return self
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.min_ = 0.0
        self.max_ = 1.0
        return self

    def transform(self, X):
        if self.data_min_ is None or self.data_max_ is None:
            return X
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_ + 1e-9)
        X_scaled = X_std * (self.max_ - self.min_) + self.min_
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SimpleClassifier:
    """SVC 대체 구현"""

    def __init__(self):
        self.threshold = 0.5
        self.is_fitted = False
        self.weights = None

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            self.is_fitted = False
            return self

        positive_samples = X[y == 1]
        negative_samples = X[y == 0]

        if len(positive_samples) > 0 and len(negative_samples) > 0:
            pos_mean = np.mean(positive_samples, axis=0)
            neg_mean = np.mean(negative_samples, axis=0)
            self.weights = pos_mean - neg_mean
            self.is_fitted = True
        else:
            self.is_fitted = False
        return self

    def predict(self, X):
        if not self.is_fitted:
            return np.random.randint(0, 2, len(X))
        scores = np.dot(X, self.weights)
        return (scores > 0).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# scikit-learn 사용 가능 여부 확인
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
    logger.info("[OK] scikit-learn 사용 가능")
except ImportError:
    SKLEARN_AVAILABLE = False
    MinMaxScaler = SimpleScaler
    SVC = SimpleClassifier
    logger.warning("[WARN] scikit-learn 사용 불가 - 단순 구현체 사용")


# === 양자 머신러닝 모델 ===
class QuantumTradingModel:
    def __init__(self, feature_dimension=QUANTUM_FEATURE_DIMENSION, timeframe="15m"):
        self.feature_dimension = feature_dimension
        self.timeframe = timeframe
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.training_history = []
        self.pattern_aware = True  # 패턴 인식 기능 활성화

    def prepare_quantum_features(self, df):
        """양자 모델용 특징 추출 (패턴 특징 포함)"""
        features = []
        df_with_indicators = add_technical_indicators(df)

        for i in range(len(df_with_indicators)):
            if i < 20:
                # 특징 개수를 5개로 일관성 있게 유지
                features.append([0.5, 0.5, 0.5, 0.5, 0.5])
                continue

            row = df_with_indicators.iloc[i]

            # RSI 특징
            rsi_feature = min(max(row['rsi'] / 100.0, 0.0), 1.0) if not np.isnan(row['rsi']) else 0.5

            # 이동평균 특징
            if not np.isnan(row['ma5']) and not np.isnan(row['ma20']) and row['ma20'] != 0:
                ma_diff = (row['ma5'] - row['ma20']) / row['ma20']
                ma_feature = min(max(ma_diff, -0.1), 0.1) * 5 + 0.5
            else:
                ma_feature = 0.5

            # 변동성 특징
            if not np.isnan(row['volatility']):
                vol_feature = min(max(row['volatility'] * 100, 0.0), 5.0) / 5.0
            else:
                vol_feature = 0.5

            # ATR 특징
            if not np.isnan(row['atr']) and row['close'] != 0:
                atr_feature = min(max(row['atr'] / row['close'], 0.0), 0.1) * 10
            else:
                atr_feature = 0.5

            # 패턴 특징 (최근 10개 봉의 상승/하락 패턴)
            pattern_feature = 0.5
            if self.pattern_aware and i >= 10:
                recent_closes = df_with_indicators['close'].iloc[i - 9:i + 1]
                pattern = []
                for j in range(1, len(recent_closes)):
                    if recent_closes.iloc[j] > recent_closes.iloc[j - 1]:
                        pattern.append(1)
                    elif recent_closes.iloc[j] < recent_closes.iloc[j - 1]:
                        pattern.append(-1)
                    else:
                        pattern.append(0)

                # 패턴 보너스 점수 계산
                current_hour = datetime.now().hour
                market_regime = detect_market_regime(df_with_indicators.iloc[:i + 1])
                pattern_bonus = get_pattern_bonus(pattern, market_regime, current_hour)
                pattern_feature = 0.5 + pattern_bonus * 0.5  # 0.5~1.0 범위

            feature_vector = [rsi_feature, ma_feature, vol_feature, atr_feature, pattern_feature]
            features.append(feature_vector)

        return np.array(features)
#----------
    def train(self, df, force_retrain=False):
        """양자 모델 학습"""
        try:
            if not force_retrain:
                saved_model, saved_scaler = load_model(self.timeframe)
                if saved_model is not None and saved_scaler is not None:
                    # 저장된 스케일러의 특징 개수 확인
                    saved_feature_dim = None
                    if hasattr(saved_scaler, 'n_features_in_'):
                        saved_feature_dim = saved_scaler.n_features_in_
                    elif hasattr(saved_scaler, 'data_min_') and saved_scaler.data_min_ is not None:
                        saved_feature_dim = len(saved_scaler.data_min_)

                    # 특징 개수가 일치하지 않으면 재학습
                    if saved_feature_dim is not None and saved_feature_dim == self.feature_dimension:
                        self.model = saved_model
                        self.scaler = saved_scaler
                        self.is_trained = True
                        logger.info(f"[SAVE] 저장된 {self.timeframe} 모델 로드 완료 (특징 개수: {saved_feature_dim})")
                        return
                    else:
                        logger.warning(
                            f"[WARN] 저장된 스케일러의 특징 개수({saved_feature_dim})와 현재 특징 개수({self.feature_dimension})가 다릅니다. 재학습을 진행합니다.")

            logger.info(f"[QUANTUM] {self.timeframe} 양자 머신러닝 모델 학습 중...")
            X = self.prepare_quantum_features(df)
            y = (df['close'].shift(-1) > df['close']).astype(int).values

            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]

            if len(X) < 20:
                logger.warning(f"[WARN] {self.timeframe} 데이터 부족 - 기본 모델 사용")
                if len(X) > 10:
                    self.scaler.fit(X)
                    X_scaled = self.scaler.transform(X)
                    self.model = SVC()
                    self.model.fit(X_scaled, y)
                    self.is_trained = True
                return

            # 스케일러 재학습 (특징 개수 일관성 유지)
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            self.model = SVC()
            self.model.fit(X_train, y_train)

            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test) if len(X_test) > 0 else 0

            logger.info(f"[OK] {self.timeframe} 모델 학습 완료 - Train: {train_score:.3f}, Test: {test_score:.3f}")
            self.is_trained = True

            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'train_score': train_score,
                'test_score': test_score,
                'data_size': len(X)
            })

            save_model(self.model, self.scaler, self.timeframe)

        except Exception as e:
            logger.error(f"[ERROR] {self.timeframe} 모델 학습 실패: {e}")
            # 에러 발생 시 기본 모델로 대체
            X = self.prepare_quantum_features(df)
            y = (df['close'].shift(-1) > df['close']).astype(int).values

            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]

            if len(X) > 10:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.model = SVC()
                self.model.fit(X_scaled, y)
                self.is_trained = True
                logger.info(f"[OK] {self.timeframe} 기본 분류기로 대체 학습 완료")
#---------------------
    def predict(self, df):
        """양자 모델 예측"""
        if not self.is_trained or self.model is None:
            logger.warning(f"[WARN] {self.timeframe} 모델이 학습되지 않음 - 기본 예측값 반환")
            return 1

        try:
            X = self.prepare_quantum_features(df)
            if len(X) == 0:
                return 1

            latest_features = X[-1].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(latest_features_scaled)[0]
            return prediction
        except Exception as e:
            logger.error(f"[ERROR] {self.timeframe} 예측 실패: {e}")
            return 1


# === 거래 관련 함수 ===
def fetch_ohlcv(timeframe="15m", limit=200):
    """OHLCV 데이터 가져오기"""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 10:
            logger.warning(f"[WARN] OHLCV 데이터 부족 ({len(ohlcv)})")
            return None
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df['ts'] = pd.to_datetime(df['ts'], unit="ms")
        return df
    except Exception as e:
        logger.error(f"[ERROR] OHLCV 데이터 조회 실패: {e}")
        return None


def get_current_price():
    """현재 가격 조회"""
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return ticker['last']
    except Exception as e:
        logger.error(f"[ERROR] 현재 가격 조회 실패: {e}")
        return None


def set_leverage(leverage):
    """레버리지 설정"""
    try:
        logger.info(f"[LEVERAGE] 레버리지 설정 시도: {leverage}x")

        # 레버리지 설정 파라미터
        params = {
            'leverage': leverage,
            'marginMode': TRADING_MODE
        }

        # 레버리지 설정
        result = exchange.set_leverage(leverage, SYMBOL, params)
        logger.info(f"[LEVERAGE] 레버리지 설정 완료: {leverage}x")
        return True
    except Exception as e:
        logger.error(f"[ERROR] 레버리지 설정 실패: {e}")
        return False


def moving_average_crossover(df, short_window=5, long_window=20):
    """이동평균 교차 신호"""
    try:
        if len(df) < long_window:
            return None
        short_ma = df['close'].rolling(window=short_window).mean()
        long_ma = df['close'].rolling(window=long_window).mean()
        if np.isnan(short_ma.iloc[-1]) or np.isnan(long_ma.iloc[-1]):
            return None
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 1
        else:
            return 0
    except Exception as e:
        logger.error(f"이동평균 신호 생성 실패: {e}")
        return None


def detect_market_regime(df):
    """시장 상태 감지"""
    if df is None or len(df) < 200:
        return "UNKNOWN"

    try:
        df_with_indicators = add_technical_indicators(df)
        ma_short = df_with_indicators['ma20']
        ma_long = df_with_indicators['close'].rolling(200).mean()

        returns = df_with_indicators['return'].dropna()
        if len(returns) < 20:
            volatility = 0.0
        else:
            volatility = returns.tail(20).std() * np.sqrt(365 * 24)

        if len(ma_short) > 0 and len(ma_long) > 0 and not np.isnan(ma_short.iloc[-1]) and not np.isnan(
                ma_long.iloc[-1]):
            trend_strength = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        else:
            trend_strength = 0.0

        rsi = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators.columns and len(
            df_with_indicators['rsi']) > 0 else 50

        if abs(trend_strength) > 0.02:
            if trend_strength > 0:
                if volatility > 0.03:
                    return "BULL_HIGH_VOL"
                else:
                    return "BULL_LOW_VOL"
            else:
                if volatility > 0.03:
                    return "BEAR_HIGH_VOL"
                else:
                    return "BEAR_LOW_VOL"
        else:
            if volatility > 0.025:
                return "SIDEWAYS_HIGH_VOL"
            else:
                return "SIDEWAYS_LOW_VOL"

    except Exception as e:
        logger.error(f"시장 Regime 감지 실패: {e}")
        return "UNKNOWN"


def get_trade_pattern(df):
    """거래 패턴 추출"""
    if df is None or len(df) < 15:
        return []
    pattern = []
    try:
        closes = df['close'].iloc[-10:]
        for i in range(1, len(closes)):
            if closes.iloc[i] > closes.iloc[i - 1]:
                pattern.append(1)
            elif closes.iloc[i] < closes.iloc[i - 1]:
                pattern.append(-1)
            else:
                pattern.append(0)
    except Exception as e:
        logger.error(f"거래 패턴 추출 실패: {e}")
        pattern = []
    return pattern


def get_account_and_position_status():
    """계정 및 포지션 상태 확인"""
    try:
        # 계정 잔고 조회
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {})
        total_balance = usdt_balance.get('total', 0.0)
        free_balance = usdt_balance.get('free', 0.0)
        used_balance = usdt_balance.get('used', 0.0)

        if PERFORMANCE_DATA['initial_balance'] == 0 and total_balance > 0:
            PERFORMANCE_DATA['initial_balance'] = total_balance

        initial_balance = PERFORMANCE_DATA['initial_balance']
        total_pnl_percent = ((total_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0

        balance_details = {
            'total': total_balance,
            'free': free_balance,
            'used': used_balance,
            'total_pnl_percent': total_pnl_percent,
            'initial_balance': initial_balance
        }

        # 포지션 정보 조회
        positions = exchange.fetch_positions([SYMBOL])
        active_positions = []

        for pos in positions:
            if float(pos.get('contracts', 0)) != 0:
                entry_price = float(pos.get('entryPrice', 0))
                size = float(pos.get('contracts', 0))
                side = pos.get('side', '').lower()  # OKX는 소문자 사용
                unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                leverage = float(pos.get('leverage', 1))

                # 현재 가격으로 PNL% 계산 (레버리지 반영)
                current_price = get_current_price()
                if current_price and entry_price > 0:
                    if side == 'long':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100 * leverage
                    else:  # short
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100 * leverage
                else:
                    pnl_percent = 0.0

                position_id = f"{pos.get('symbol')}_{side}_{entry_price}"

                active_positions.append({
                    "position_id": position_id,
                    "symbol": pos.get('symbol'),
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "pnl": unrealized_pnl,
                    "floating_pnl_percent": pnl_percent,
                    "current_price": current_price,
                    "leverage": leverage,
                    "timestamp": datetime.now().isoformat()
                })

        return balance_details, active_positions

    except Exception as e:
        logger.error(f"[ERROR] 잔고/포지션 정보 조회 실패: {e}")
        return None, []


def display_account_info(balance, positions, current_price):
    """계정 정보 표시"""
    try:
        print("\n" + "=" * 80)
        print("[CHART] REAL-TIME ACCOUNT STATUS")
        print("=" * 80)

        # 잔고 정보
        if balance:
            print(f"[MONEY] USDT Balance: {balance['total']:.2f} USDT")
            print(f"        ├─ Available: {balance['free']:.2f} USDT")
            print(f"        ├─ In Use: {balance['used']:.2f} USDT")
            print(f"        └─ Total PnL: {balance['total_pnl_percent']:+.2f}%")

        # 현재 가격
        if current_price:
            print(f"[UP] Current BTC Price: {current_price:.2f} USDT")

        # 포지션 정보
        if positions:
            print(f"[TARGET] Active Positions: {len(positions)}")
            for i, pos in enumerate(positions, 1):
                pnl_color = "[UP]" if pos['pnl'] > 0 else "[DOWN]"
                print(f"        {i}. {pos['side'].upper()} {pos['size']:.3f} contracts")
                print(f"             ├─ Entry: {pos['entry_price']:.2f} USDT")
                print(f"             ├─ Current: {pos['current_price']:.2f} USDT")
                print(f"             ├─ PnL: {pos['pnl']:+.3f} USDT {pnl_color}")
                print(f"             ├─ PnL%: {pos['floating_pnl_percent']:+.2f}% {pnl_color}")
                print(f"             └─ Leverage: {pos['leverage']}x")
        else:
            print("[TARGET] Active Positions: None")

        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"[ERROR] 계정 정보 표시 실패: {e}")


def generate_clordid():
    """주문 ID 생성"""
    timestamp = str(int(time.time() * 1000))[-8:]
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"{timestamp}{random_str}"


def execute_order(params):
    """주문 실행"""
    try:
        if 'params' not in params:
            params['params'] = {}
        params['params']['tdMode'] = TRADING_MODE

        return exchange.create_order(**params)
    except Exception as e:
        logger.error(f"[ERROR] 주문 실행 실패: {e}")
        raise e


def place_order(signal: int, amount: float) -> bool:
    """주문 placement"""
    try:
        side = 'buy' if signal == 1 else 'sell'
        logger.info(f"[LAUNCH] 신규 주문: {side.upper()} / {amount:.3f} 계약 (모드: {TRADING_MODE}, 레버리지: {LEVERAGE}x)")

        params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": amount,
            "params": {
                "tdMode": TRADING_MODE,
            }
        }

        params["params"]["clOrdId"] = generate_clordid()

        result = execute_order(params)
        logger.info(f"[OK] 주문 성공: {result['id']}")

        # 주문 성공 후 트레일링 스탑 초기화
        time.sleep(2)  # 잠시 대기 후 포지션 확인
        balance, positions = get_account_and_position_status()
        if positions:
            position = positions[0]
            current_price = get_current_price()
            if current_price:
                initialize_trailing_stop(
                    position['position_id'],
                    position['entry_price'],
                    position['side'],
                    current_price
                )

        return True
    except Exception as e:
        logger.error(f"[ERROR] 주문 실패: {e}")
        return False


def close_position(position: Dict, amount: float, description: str) -> bool:
    """포지션 청산"""
    try:
        side = "sell" if position["side"] == "long" else "buy"
        logger.info(f"[LOCK] {description} 실행: {position['side'].upper()} / {amount:.3f} 계약")

        params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": amount,
            "params": {
                "tdMode": TRADING_MODE,
            }
        }

        params["params"]["clOrdId"] = generate_clordid()

        result = execute_order(params)
        logger.info(f"[OK] 청산 성공: {result['id']}")

        # 트레일링 스탑 제거
        remove_trailing_stop(position['position_id'])

        pnl_usdt = position['pnl']
        pnl_percent = position['floating_pnl_percent']
        market_regime = detect_market_regime(fetch_ohlcv(timeframe=TIMEFRAME_15M, limit=100))
        hour_of_day = datetime.now().hour

        record_trade(position, pnl_usdt, pnl_percent, market_regime, hour_of_day)
        return True
    except Exception as e:
        logger.error(f"[ERROR] 청산 실패: {e}")
        return False


def check_profit_loss_conditions(position):
    """수익/손실 조건 체크"""
    if not position:
        return False

    pnl_percent = position['floating_pnl_percent']

    if pnl_percent >= TAKE_PROFIT_PERCENT:
        logger.info(f"[TARGET] 익절 조건 달성: {pnl_percent:.2f}% ≥ {TAKE_PROFIT_PERCENT}%")
        return close_position(position, position['size'], f"익절 ({pnl_percent:.2f}%)")

    elif pnl_percent <= STOP_LOSS_PERCENT:
        logger.info(f"[WARN] 손절 조건 달성: {pnl_percent:.2f}% ≤ {STOP_LOSS_PERCENT}%")
        return close_position(position, position['size'], f"손절 ({pnl_percent:.2f}%)")

    return False


def check_trailing_stop_conditions():
    """트레일링 스탑 조건 체크"""
    try:
        current_price = get_current_price()
        if not current_price:
            return False

        positions_to_close = check_all_trailing_stops(current_price)

        if positions_to_close:
            balance, positions = get_account_and_position_status()
            for position_id in positions_to_close:
                # 해당 포지션 찾기
                position_to_close = None
                for pos in positions:
                    if pos['position_id'] == position_id:
                        position_to_close = pos
                        break

                if position_to_close:
                    logger.info(f"[TRAILING] 트레일링 스탑 청산 실행: {position_id}")
                    close_position(position_to_close, position_to_close['size'],
                                   f"트레일링 스탑 청산 ({TRAILING_STOP_PERCENT}%)")
                    return True

        return False
    except Exception as e:
        logger.error(f"[ERROR] 트레일링 스탑 조건 체크 실패: {e}")
        return False


# === 수정된 거래 기록 함수 ===
def record_trade(position_data, pnl_usdt, pnl_percent, market_regime, hour_of_day, signal_accuracy=None,
                 trade_pattern=None):
    """거래 기록 저장 (패턴 학습 포함)"""
    trade_record = {
        'timestamp': datetime.now().isoformat(),
        'symbol': position_data.get('symbol', SYMBOL),
        'side': position_data.get('side', 'UNKNOWN'),
        'pnl_usdt': pnl_usdt,
        'pnl_percent': pnl_percent,
        'market_regime': market_regime,
        'hour_of_day': hour_of_day,
        'signal_accuracy': signal_accuracy,
        'trade_pattern': trade_pattern,
        'duration_minutes': position_data.get('duration_minutes', 0)
    }

    # 기존 성과 데이터 업데이트
    PERFORMANCE_DATA['total_trades'] += 1
    PERFORMANCE_DATA['total_pnl'] += pnl_usdt

    if pnl_usdt > 0:
        PERFORMANCE_DATA['winning_trades'] += 1

    PERFORMANCE_DATA['recent_trades'].append(trade_record)

    # 패턴 학습 업데이트
    update_pattern_learning(trade_record)

    # 기존 성과 분석 데이터 업데이트
    if trade_pattern:
        pattern_key = str(trade_pattern)
        if pattern_key not in PERFORMANCE_DATA['similar_trade_patterns']:
            PERFORMANCE_DATA['similar_trade_patterns'][pattern_key] = {
                'count': 0,
                'total_pnl': 0.0,
                'winning_trades': 0
            }

        PERFORMANCE_DATA['similar_trade_patterns'][pattern_key]['count'] += 1
        PERFORMANCE_DATA['similar_trade_patterns'][pattern_key]['total_pnl'] += pnl_usdt
        if pnl_usdt > 0:
            PERFORMANCE_DATA['similar_trade_patterns'][pattern_key]['winning_trades'] += 1

    if signal_accuracy is not None:
        PERFORMANCE_DATA['signal_accuracy_history'].append(signal_accuracy)

    regime = market_regime
    if regime not in PERFORMANCE_DATA['market_regime_performance']:
        PERFORMANCE_DATA['market_regime_performance'][regime] = {'trades': 0, 'total_pnl': 0.0, 'wins': 0}

    PERFORMANCE_DATA['market_regime_performance'][regime]['trades'] += 1
    PERFORMANCE_DATA['market_regime_performance'][regime]['total_pnl'] += pnl_usdt
    if pnl_usdt > 0:
        PERFORMANCE_DATA['market_regime_performance'][regime]['wins'] += 1

    hour_key = f"hour_{hour_of_day}"
    if hour_key not in PERFORMANCE_DATA['time_based_performance']:
        PERFORMANCE_DATA['time_based_performance'][hour_key] = {'trades': 0, 'total_pnl': 0.0, 'wins': 0}

    PERFORMANCE_DATA['time_based_performance'][hour_key]['trades'] += 1
    PERFORMANCE_DATA['time_based_performance'][hour_key]['total_pnl'] += pnl_usdt
    if pnl_usdt > 0:
        PERFORMANCE_DATA['time_based_performance'][hour_key]['wins'] += 1

    log_performance(f"{pnl_usdt},{pnl_percent},{regime},{hour_of_day}")
    save_trade_history()


def find_similar_profitable_trade(current_pattern):
    """유사한 수익성 거래 패턴 찾기"""
    if not current_pattern or not PERFORMANCE_DATA['similar_trade_patterns']:
        return None

    current_key = str(current_pattern)
    similar_patterns = {}

    for pattern_key, stats in PERFORMANCE_DATA['similar_trade_patterns'].items():
        if stats['count'] >= 2:
            win_rate = stats['winning_trades'] / stats['count']
            avg_pnl = stats['total_pnl'] / stats['count']

            if win_rate > 0.6 or avg_pnl > 0:
                similar_patterns[pattern_key] = {
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'count': stats['count']
                }

    if similar_patterns:
        best_pattern = max(similar_patterns.items(),
                           key=lambda x: (x[1]['win_rate'], x[1]['avg_pnl']))
        return best_pattern[0], best_pattern[1]

    return None


def performance_analysis():
    """성과 분석"""
    if len(PERFORMANCE_DATA['recent_trades']) < 5:
        return

    recent_trades = list(PERFORMANCE_DATA['recent_trades'])
    win_rate = len([t for t in recent_trades if t['pnl_usdt'] > 0]) / len(recent_trades)
    avg_win = np.mean([t['pnl_usdt'] for t in recent_trades if t['pnl_usdt'] > 0]) if any(
        t['pnl_usdt'] > 0 for t in recent_trades) else 0
    avg_loss = np.mean([t['pnl_usdt'] for t in recent_trades if t['pnl_usdt'] < 0]) if any(
        t['pnl_usdt'] < 0 for t in recent_trades) else 0

    logger.info(f"[CHART] 성과 분석 - 승률: {win_rate:.2%}, 평균 이익: {avg_win:.4f}, 평균 손실: {avg_loss:.4f}")

    if PERFORMANCE_DATA['signal_accuracy_history']:
        accuracy_rate = sum(PERFORMANCE_DATA['signal_accuracy_history']) / len(
            PERFORMANCE_DATA['signal_accuracy_history'])
        logger.info(f"[TARGET] 신호 정확도: {accuracy_rate:.2%}")

        global INTERVAL_WAITING
        if accuracy_rate < 0.5:
            INTERVAL_WAITING = 600
            logger.info("[WARN] 신호 정확도 낮음 - 대기 시간 증가")
        else:
            INTERVAL_WAITING = 300


def get_optimal_trading_time():
    """최적 거래 시간 분석"""
    if not PERFORMANCE_DATA['time_based_performance']:
        return None

    best_hour = None
    best_performance = -float('inf')

    for hour_key, stats in PERFORMANCE_DATA['time_based_performance'].items():
        if stats['trades'] >= 3:
            avg_pnl = stats['total_pnl'] / stats['trades']
            if avg_pnl > best_performance:
                best_performance = avg_pnl
                best_hour = int(hour_key.split('_')[1])

    return best_hour


# === 메인 함수 ===
def main():
    global PERFORMANCE_DATA, LEVERAGE, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT

    # 명령줄 인자 처리
    parser = argparse.ArgumentParser(description='양자 트레이딩 봇')
    parser.add_argument('--mode', type=str, default='trade', choices=['trade', 'backtest', 'train'],
                        help='실행 모드: trade(실거래), backtest(백테스트), train(학습만)')
    parser.add_argument('--retrain', action='store_true', help='모델 재학습')
    parser.add_argument('--analyze-patterns', action='store_true', help='패턴 분석만 실행')
    parser.add_argument('--leverage', type=int, default=100, help='레버리지 설정 (기본값: 100)')
    args = parser.parse_args()

    # 레버리지 설정 적용
    LEVERAGE = args.leverage
    TAKE_PROFIT_PERCENT = 3000.0 / LEVERAGE
    STOP_LOSS_PERCENT = -5000.0 / LEVERAGE

    if args.analyze_patterns:
        load_pattern_learning()
        analyze_pattern_performance()
        sys.exit(0)

    if args.mode == 'train' or args.retrain:
        logger.info("[TARGET] 학습 모드 실행")
        df_15m = fetch_ohlcv(timeframe=TIMEFRAME_15M, limit=1000)
        df_4h = fetch_ohlcv(timeframe=TIMEFRAME_4H, limit=1000)
        if df_15m is not None and df_4h is not None:
            quantum_model_15m = QuantumTradingModel(timeframe="15m")
            quantum_model_4h = QuantumTradingModel(timeframe="4h")
            quantum_model_15m.train(df_15m, force_retrain=True)
            quantum_model_4h.train(df_4h, force_retrain=True)
        sys.exit(0)

    # 메인 트레이딩 루프
    PERFORMANCE_DATA = load_trade_history()
    load_trailing_stops()
    load_pattern_learning()  # 패턴 학습 데이터 로드

    quantum_model_15m = QuantumTradingModel(timeframe="15m")
    quantum_model_4h = QuantumTradingModel(timeframe="4h")

    error_count, max_errors = 0, 5
    time_to_wait = 0
    waiting_mode = False

    logger.info("[LAUNCH] Ubuntu OKX 트레이딩 봇 시작 (패턴 학습 활성화)")
    logger.info(f"[CHART] 이전 거래 기록: {PERFORMANCE_DATA['total_trades']}회 거래")
    logger.info(f"[LEARN] 학습된 패턴: {len(PATTERN_LEARNING_DATA['profitable_patterns'])}개 수익성 패턴")
    logger.info(f"[LEARN] 패턴 가중치: {len(PATTERN_LEARNING_DATA['pattern_weights'])}개 패턴에 가중치 적용")
    logger.info(f"[LEVERAGE] 설정 레버리지: {LEVERAGE}x")
    logger.info(f"[LEVERAGE] 조정된 익절: {TAKE_PROFIT_PERCENT:.2f}%")
    logger.info(f"[LEVERAGE] 조정된 손절: {STOP_LOSS_PERCENT:.2f}%")
    logger.info(f"[TOOL] 거래 모드: {TRADING_MODE}")
    logger.info(f"[TRAILING] 트레일링 스탑: {TRAILING_STOP_PERCENT}%")
    logger.info(f"[FOLDER] 데이터 디렉토리: {DATA_DIR}")
    logger.info(f"[PC] 플랫폼: {'Ubuntu/Linux' if IS_LINUX else 'Windows'}")
    logger.info(f"[ROBOT] 머신러닝: {'scikit-learn 사용' if SKLEARN_AVAILABLE else '단순 구현체 사용'}")
    logger.info(f"[TIME] 타임프레임: 15분봉 & 4시간봉")
    logger.info(f"[TARGET] 신호 일치 임계값: {SIGNAL_MATCH_THRESHOLD:.0%}")

    # API 인증 상태 표시
    if API_KEY and API_SECRET and API_PASSPHRASE:
        logger.info("[OK] API 인증: 실거래 모드")
        # 레버리지 설정
        if not set_leverage(LEVERAGE):
            logger.warning("[WARN] 레버리지 설정 실패, 기본값 사용")
    else:
        logger.info("[WARN] API 인증: 샌드박스 모드 (공개 데이터만 조회 가능)")

    # 초기 패턴 분석 실행
    analyze_pattern_performance()

    while True:
        try:
            time.sleep(time_to_wait)
            start_time = time.time()

            # 주기적 모델 재학습 (6시간마다)
            current_hour = datetime.now().hour
            if current_hour % 6 == 0:  # 6시간마다 재학습
                retrain_models_with_patterns()
                analyze_pattern_performance()

            # 현재 가격 조회 (트레일링 스탑 체크용)
            current_price = get_current_price()

            # 계정 정보 조회 및 표시
            balance, positions = get_account_and_position_status()
            display_account_info(balance, positions, current_price)

            if current_price:
                # 트레일링 스탑 조건 체크
                trailing_stop_triggered = check_trailing_stop_conditions()
                if trailing_stop_triggered:
                    logger.info("[TRAILING] 트레일링 스탑 청산 실행됨")
                    time.sleep(5)  # 청산 후 잠시 대기

            # 15분봉과 4시간봉 데이터 조회
            df_15m = fetch_ohlcv(timeframe=TIMEFRAME_15M, limit=CANDLE_LIMIT)
            df_4h = fetch_ohlcv(timeframe=TIMEFRAME_4H, limit=CANDLE_LIMIT)

            if df_15m is None or df_4h is None:
                error_count += 1
                time_to_wait = INTERVAL_NORMAL
                if error_count >= max_errors:
                    break
                continue

            # 모델 학습 (필요시)
            if not quantum_model_15m.is_trained:
                quantum_model_15m.train(df_15m)
            if not quantum_model_4h.is_trained:
                quantum_model_4h.train(df_4h)

            # 양자 예측
            quantum_signal_15m = quantum_model_15m.predict(df_15m)
            quantum_signal_4h = quantum_model_4h.predict(df_4h)

            quantum_text_15m = '상승(BUY)' if quantum_signal_15m == 1 else '하락(SELL)'
            quantum_text_4h = '상승(BUY)' if quantum_signal_4h == 1 else '하락(SELL)'

            logger.info(f"[QUANTUM] 15M 양자 예측: {quantum_text_15m}")
            logger.info(f"[QUANTUM] 4H 양자 예측: {quantum_text_4h}")

            # 기존 기술적 분석 신호
            signal_15m = moving_average_crossover(df_15m)
            signal_4h = moving_average_crossover(df_4h)

            if signal_15m is None or signal_4h is None:
                logger.warning("[ERROR] 신호 생성 실패 - 기본 전략 사용")
                signal_15m = 1
                signal_4h = 1

            # 신호 일치율 계산
            signals_15m_match = (signal_15m == quantum_signal_15m)
            signals_4h_match = (signal_4h == quantum_signal_4h)
            total_match_rate = (int(signals_15m_match) + int(signals_4h_match)) / 2.0

            logger.info(f"[RADAR] 15M 신호 일치: {'[OK] 일치' if signals_15m_match else '[ERROR] 불일치'}")
            logger.info(f"[RADAR] 4H 신호 일치: {'[OK] 일치' if signals_4h_match else '[ERROR] 불일치'}")
            logger.info(f"[TARGET] 전체 신호 일치율: {total_match_rate:.1%}")

            # 시장 상태 분석
            market_regime = detect_market_regime(df_15m)
            logger.info(f"[TARGET] 현재 시장 Regime: {market_regime}")

            current_pattern = get_trade_pattern(df_15m)
            similar_profitable_trade = find_similar_profitable_trade(current_pattern)

            if similar_profitable_trade:
                logger.info(f"[DICE] 유사한 수익성 패턴 발견: 승률 {similar_profitable_trade[1]['win_rate']:.1%}")

            performance_analysis()

            active_position = positions[0] if positions else None
            total_pnl_percent = balance.get('total_pnl_percent', 0) if balance else 0

            if active_position:
                logger.info(
                    f"[PACKAGE] 포지션 감지: {active_position['side']}, PNL: {active_position['pnl']:.3f} USDT ({active_position['floating_pnl_percent']:.2f}%)")

                # 트레일링 스탑이 없는 포지션에 대해 초기화
                if active_position['position_id'] not in TRAILING_STOPS and current_price:
                    initialize_trailing_stop(
                        active_position['position_id'],
                        active_position['entry_price'],
                        active_position['side'],
                        current_price
                    )

                if total_pnl_percent <= EMERGENCY_LIQUIDATION_THRESHOLD:
                    logger.info(f"[FIRE] 긴급 전체 청산! 전체 자본 손실: {total_pnl_percent:.2f}%")
                    close_position(active_position, active_position['size'], "긴급 전체 청산")

                elif check_profit_loss_conditions(active_position):
                    logger.info("[SYNC] 포지션 변경 후 상태 업데이트 중...")
                    time.sleep(3)
                    balance, positions = get_account_and_position_status()
                else:
                    logger.info("[OK] 현재 포지션 유지 중")

            else:
                logger.info("[NEW] 신규 포지션 진입 검토 중...")

                # 신호 일치율이 임계값 이상일 때만 매수
                should_enter = (total_match_rate >= SIGNAL_MATCH_THRESHOLD)

                if should_enter:
                    logger.info(f"[OK] 신호 일치율 {total_match_rate:.1%} ≥ {SIGNAL_MATCH_THRESHOLD:.0%} - 진입 조건 충족")

                    # 최종 신호 결정 (양자 예측 우선)
                    final_signal = 1 if (quantum_signal_15m + quantum_signal_4h) >= 1 else 0
                    final_signal_text = '상승(BUY)' if final_signal == 1 else '하락(SELL)'

                    best_hour = get_optimal_trading_time()
                    current_hour = datetime.now().hour
                    if best_hour is not None:
                        time_match = (current_hour == best_hour)
                        time_info = f" ({best_hour}시 - {'[OK] 최적시간' if time_match else '[WARN] 일반시간'})"
                    else:
                        time_info = ""

                    pattern_info = ""
                    if similar_profitable_trade:
                        pattern_info = f" [유사패턴 승률: {similar_profitable_trade[1]['win_rate']:.1%}]"

                    logger.info(f"[TARGET] 진입 결정: {final_signal_text}{time_info}{pattern_info}")

                    if place_order(final_signal, CONTRACT_AMOUNT):
                        logger.info("[SYNC] 신규 주문 후 상태 업데이트 중...")
                        time.sleep(3)
                        balance, positions = get_account_and_position_status()

                        # 신호 정확도 기록
                        signal_accuracy = (final_signal == 1 and df_15m['close'].iloc[-1] < df_15m['close'].iloc[-2]) or \
                                          (final_signal == 0 and df_15m['close'].iloc[-1] > df_15m['close'].iloc[-2])
                        PERFORMANCE_DATA['signal_accuracy_history'].append(signal_accuracy)
                else:
                    logger.info(
                        f"[PAUSE] 신호 불일치 ({total_match_rate:.1%} < {SIGNAL_MATCH_THRESHOLD:.0%}) - {INTERVAL_WAITING}초 대기")
                    waiting_mode = True

            if waiting_mode:
                monitoring_interval = INTERVAL_WAITING
                waiting_mode = False
            else:
                monitoring_interval = INTERVAL_ACTIVE if positions else INTERVAL_NORMAL

            error_count = 0
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, monitoring_interval - elapsed_time)

            logger.info(f"[CLOCK] 다음 실행까지 {int(time_to_wait)}초 대기")

        except KeyboardInterrupt:
            logger.info("\n[STOP] 사용자에 의해 프로그램 종료")
            save_trade_history()
            save_trailing_stops()
            save_pattern_learning()  # 패턴 학습 데이터 저장
            break
        except Exception as e:
            logger.error(f"[FIRE] 메인 루프 오류: {e}")
            traceback.print_exc()
            error_count += 1
            if error_count >= max_errors:
                save_trade_history()
                save_trailing_stops()
                save_pattern_learning()  # 패턴 학습 데이터 저장
                break
            time_to_wait = INTERVAL_NORMAL


if __name__ == "__main__":
    main()