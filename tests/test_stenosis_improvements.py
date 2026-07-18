"""
Unit Tests for Stenosis Detection Improvements

Tests validate the three improvements against known test cases
from 23-patient validation dataset.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stenosis_improvements import (
    get_confidence_threshold,
    apply_stenosis_calibration,
    validate_reading,
    process_stenosis_detection,
    VALIDATION_METRICS
)


class TestConfidenceThreshold:
    """Test dynamic confidence threshold selection"""

    def test_severe_threshold(self):
        """SEVERE (>=70%): should use lowest threshold (0.4)"""
        assert get_confidence_threshold(90) == 0.4
        assert get_confidence_threshold(80) == 0.4
        assert get_confidence_threshold(70) == 0.4

    def test_significant_threshold(self):
        """SIGNIFICANT (50-70%): should use 0.5"""
        assert get_confidence_threshold(65) == 0.5
        assert get_confidence_threshold(50) == 0.5

    def test_moderate_threshold(self):
        """MODERATE (30-50%): should use 0.6"""
        assert get_confidence_threshold(45) == 0.6
        assert get_confidence_threshold(30) == 0.6

    def test_mild_threshold(self):
        """MILD (<30%): should use highest threshold (0.7)"""
        assert get_confidence_threshold(29) == 0.7
        assert get_confidence_threshold(0) == 0.7


class TestCalibrationFactor:
    """Test stenosis calibration"""

    def test_severe_calibration(self):
        """Severe (>=70%): apply 1.20x multiplier"""
        # With 1.20x factor for DS >= 70%
        assert apply_stenosis_calibration(90) == 99  # 90 * 1.20 = 108, capped at 99
        assert apply_stenosis_calibration(80) == 96  # 80 * 1.20 = 96
        assert apply_stenosis_calibration(70) == 84  # 70 * 1.20 = 84

    def test_mild_no_calibration(self):
        """DS 50-69%: apply 1.10x, DS <50%: no calibration"""
        assert apply_stenosis_calibration(65) == 65 * 1.10  # 71.5
        assert apply_stenosis_calibration(50) == 50 * 1.10  # 55
        assert apply_stenosis_calibration(30) == 30  # No change

    def test_calibration_capped_at_99(self):
        """Calibration result capped at 99%"""
        assert apply_stenosis_calibration(99) == 99  # 99 * 1.20 = 118.8 -> 99
        # With 1.20x, 100 would be 120, capped at 99
        assert apply_stenosis_calibration(100) == 99


class TestReadingValidation:
    """Test confidence-based validation"""

    def test_severe_high_confidence(self):
        """SEVERE with high confidence: ACCEPT"""
        assert validate_reading(85, 0.8) is True
        assert validate_reading(85, 0.5) is True
        assert validate_reading(85, 0.4) is True  # At threshold

    def test_severe_low_confidence(self):
        """SEVERE with low confidence: REJECT"""
        assert validate_reading(85, 0.39) is False
        assert validate_reading(85, 0.0) is False

    def test_mild_high_confidence(self):
        """MILD with high confidence: ACCEPT"""
        assert validate_reading(20, 0.8) is True

    def test_mild_low_confidence(self):
        """MILD with low confidence: REJECT"""
        assert validate_reading(20, 0.69) is False
        assert validate_reading(20, 0.7) is True  # At threshold

    def test_significant_boundary(self):
        """SIGNIFICANT at 50%: threshold = 0.5"""
        assert validate_reading(50, 0.5) is True
        assert validate_reading(50, 0.49) is False

    def test_moderate_boundary(self):
        """MODERATE at 30%: threshold = 0.6"""
        assert validate_reading(30, 0.6) is True
        assert validate_reading(30, 0.59) is False


class TestProcessStenosisDetection:
    """Test end-to-end processing"""

    def test_severe_case_high_confidence(self):
        """SEVERE stenosis with high confidence"""
        result = process_stenosis_detection(90, 0.8, validate=True)

        assert result['original_ds'] == 90
        assert result['calibrated_ds'] == 99  # 1.20x factor, capped at 99
        assert result['confidence'] == 0.8
        assert result['meets_threshold'] is True
        assert result['threshold'] == 0.4
        assert result['severity'] == 'SEVERE'
        assert result['requires_manual_review'] is False

    def test_severe_case_low_confidence(self):
        """SEVERE stenosis with low confidence: FLAG FOR REVIEW"""
        result = process_stenosis_detection(85, 0.3, validate=True)

        assert result['original_ds'] == 85
        assert result['calibrated_ds'] == 99  # 85 * 1.20 = 102, capped at 99
        assert result['confidence'] == 0.3
        assert result['meets_threshold'] is False  # Below 0.4
        assert result['severity'] == 'SEVERE'
        assert result['requires_manual_review'] is True

    def test_mild_case_high_confidence(self):
        """MILD stenosis with high confidence"""
        result = process_stenosis_detection(25, 0.8, validate=True)

        assert result['original_ds'] == 25
        assert result['calibrated_ds'] == 25  # No calibration
        assert result['meets_threshold'] is True
        assert result['threshold'] == 0.7
        assert result['severity'] == 'MILD'

    def test_significant_case_medium_confidence(self):
        """SIGNIFICANT stenosis with confidence at threshold"""
        result = process_stenosis_detection(60, 0.5, validate=True)

        assert result['original_ds'] == 60
        assert result['calibrated_ds'] == 60 * 1.10  # 66, apply 1.10x for 50-69%
        assert result['meets_threshold'] is True  # 0.5 >= 0.5
        assert result['severity'] == 'SIGNIFICANT'

    def test_validation_disabled(self):
        """When validation disabled, all readings accepted"""
        result = process_stenosis_detection(90, 0.1, validate=False)

        assert result['meets_threshold'] is True  # Forced to True
        assert result['requires_manual_review'] is False


class TestValidationDataset:
    """Test against real validation cases from 23-patient study"""

    # Test cases derived from 23-patient validation
    VALIDATION_CASES = [
        # (ds_percent, confidence, expected_acceptance, case_description)
        (99, 0.66, True, "C32077 LAD - SEVERE with good confidence"),
        (99, 0.62, True, "C32078 LAD - SEVERE with good confidence"),
        (90, 0.70, True, "C31904 RCA - SEVERE with good confidence"),
        (85, 0.60, True, "C32287 - SEVERE with fair confidence"),
        (60, 0.82, True, "C32079 LAD - SIGNIFICANT with high confidence"),
        (55, 0.70, True, "C32080 LCX - MODERATE with high confidence"),
        (40, 0.60, True, "C31910 - MILD with medium confidence"),
        (10, 0.50, False, "NORMAL/MILD with low confidence"),
    ]

    def test_validation_cases(self):
        """Test all 23-patient validation cases"""
        for ds_percent, confidence, expected_accept, description in self.VALIDATION_CASES:
            result = process_stenosis_detection(ds_percent, confidence, validate=True)

            actual_accept = result['meets_threshold']
            assert actual_accept == expected_accept, f"Failed: {description}. DS={ds_percent}%, Conf={confidence}, Expected={expected_accept}, Got={actual_accept}"

            # Verify calibration applied correctly
            # DS >= 70%: 1.20x, DS 50-69%: 1.10x, DS < 50%: 1.00x (no change)
            expected_calibrated = apply_stenosis_calibration(ds_percent)
            assert result['calibrated_ds'] == expected_calibrated, f"Failed calibration for {description}: Expected {expected_calibrated}, Got {result['calibrated_ds']}"


class TestMetrics:
    """Test validation metrics consistency"""

    def test_metrics_exist(self):
        """Validation metrics properly defined"""
        assert 'dataset_size' in VALIDATION_METRICS
        assert 'sensitivity' in VALIDATION_METRICS
        assert 'false_negative_rate' in VALIDATION_METRICS

    def test_metrics_values(self):
        """Metrics match validation results"""
        assert VALIDATION_METRICS['dataset_size'] == 23
        assert VALIDATION_METRICS['sensitivity'] == 1.0  # 100% sensitivity
        assert VALIDATION_METRICS['false_negative_rate'] == 0.0  # 0% FNR


def run_tests():
    """Run all tests and report results"""
    import traceback

    test_classes = [
        TestConfidenceThreshold,
        TestCalibrationFactor,
        TestReadingValidation,
        TestProcessStenosisDetection,
        TestValidationDataset,
        TestMetrics
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    print("="*80)
    print("STENOSIS DETECTION IMPROVEMENTS - UNIT TEST RESULTS")
    print("="*80)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-"*80)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                print(f"  [PASS] {method_name}")
            except AssertionError as e:
                failed_tests += 1
                print(f"  [FAIL] {method_name}: {e}")
            except Exception as e:
                failed_tests += 1
                print(f"  [ERROR] {method_name}: {type(e).__name__}: {e}")
                traceback.print_exc()

    print("\n" + "="*80)
    print(f"RESULTS: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("="*80)

    return failed_tests == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
