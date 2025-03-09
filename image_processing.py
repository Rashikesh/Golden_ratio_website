import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch # type: ignore

class GoldenRatioFaceAnalyzer:
    def __init__(self):
        # Golden ratio constant
        self.GOLDEN_RATIO = 1.618
        
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        # You'll need to download this file from dlib's website
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Define landmark indices for key facial features
        self.FACE_INDICES = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_lips': list(range(48, 60)),
            'inner_lips': list(range(60, 68))
        }
        
    def detect_landmarks(self, image_path):
        """Detect facial landmarks in an image"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            raise ValueError("No faces detected in the image")
        
        # Use the first face found
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        
        # Convert landmarks to numpy array
        landmarks_points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))
            
        return np.array(landmarks_points), img
    
    def calculate_ratio(self, distance1, distance2):
        """Calculate ratio between two distances"""
        # Ensure we're dividing the larger by the smaller
        if distance1 > distance2:
            return distance1 / distance2
        else:
            return distance2 / distance1
    
    def get_deviation(self, ratio):
        """Calculate deviation from golden ratio as percentage"""
        return abs(ratio - self.GOLDEN_RATIO) / self.GOLDEN_RATIO * 100
    
    def analyze_facial_proportions(self, landmarks):
        """Analyze various facial proportions based on golden ratio"""
        results = {}
        
        # Extract key points
        right_eye_center = np.mean(landmarks[self.FACE_INDICES['right_eye']], axis=0)
        left_eye_center = np.mean(landmarks[self.FACE_INDICES['left_eye']], axis=0)
        eye_distance = distance.euclidean(right_eye_center, left_eye_center)
        
        nose_tip = landmarks[33]  # Central point of the nose tip
        
        top_lip = landmarks[51]  # Central point of the top lip
        bottom_lip = landmarks[57]  # Central point of the bottom lip
        lip_center = np.mean([top_lip, bottom_lip], axis=0)
        
        chin = landmarks[8]  # Bottom of the chin
        
        face_top = landmarks[27]  # Top of the nose bridge
        face_height = distance.euclidean((face_top[0], face_top[1]), (chin[0], chin[1]))
        
        # Face width at cheekbones (approximate)
        face_width = distance.euclidean(landmarks[1], landmarks[15])
        
        # Calculate relevant distances
        eye_to_lip = distance.euclidean(np.mean([right_eye_center, left_eye_center], axis=0), lip_center)
        lip_to_chin = distance.euclidean(lip_center, chin)
        
        nose_to_lip = distance.euclidean(nose_tip, lip_center)
        eye_to_nose = distance.euclidean(np.mean([right_eye_center, left_eye_center], axis=0), nose_tip)
        
        # Calculate ratios
        face_ratio = face_height / face_width
        results['face_height_to_width'] = {
            'ratio': face_ratio,
            'deviation': self.get_deviation(face_ratio)
        }
        
        eye_lip_chin_ratio = eye_to_lip / lip_to_chin
        results['eye_to_lip_vs_lip_to_chin'] = {
            'ratio': eye_lip_chin_ratio,
            'deviation': self.get_deviation(eye_lip_chin_ratio)
        }
        
        eye_nose_lip_ratio = eye_to_nose / nose_to_lip
        results['eye_to_nose_vs_nose_to_lip'] = {
            'ratio': eye_nose_lip_ratio,
            'deviation': self.get_deviation(eye_nose_lip_ratio)
        }
        
        width_to_eye_ratio = face_width / eye_distance
        results['face_width_to_eye_distance'] = {
            'ratio': width_to_eye_ratio,
            'ideal': 2.618,  # 1 + phi
            'deviation': abs(width_to_eye_ratio - 2.618) / 2.618 * 100
        }
        
        return results
    
    def calculate_overall_score(self, results):
        """Calculate an overall golden ratio score"""
        deviations = [result['deviation'] for result in results.values()]
        average_deviation = np.mean(deviations)
        
        # Convert to a 0-100 score (lower deviation = higher score)
        max_acceptable_deviation = 30  # 30% deviation considered maximum
        score = max(0, 100 - (average_deviation * 100 / max_acceptable_deviation))
        return score
    
    def visualize_analysis(self, image, landmarks, results, score):
        """Create a visualization of the analysis"""
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Display the image with landmarks
        ax1.imshow(img_rgb)
        for feature, indices in self.FACE_INDICES.items():
            ax1.plot(landmarks[indices, 0], landmarks[indices, 1], 'o-', markersize=2, linewidth=1)
            
        ax1.set_title("Facial Landmarks")
        ax1.axis('off')
        
        # Display the results as a table
        ax2.axis('tight')
        ax2.axis('off')
        
        table_data = []
        for feature, data in results.items():
            table_data.append([
                feature.replace('_', ' ').title(),
                f"{data['ratio']:.3f}",
                f"{self.GOLDEN_RATIO:.3f}",
                f"{data['deviation']:.2f}%"
            ])
        
        table = ax2.table(
            cellText=table_data,
            colLabels=["Measurement", "Actual Ratio", "Golden Ratio", "Deviation"],
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Add overall score
        ax2.text(
            0.5, 0.9, 
            f"Golden Ratio Score: {score:.1f}/100", 
            horizontalalignment='center',
            fontsize=14,
            transform=ax2.transAxes,
            bbox=dict(facecolor='gold', alpha=0.2)
        )
        
        plt.tight_layout()
        return fig
    
    def analyze_image(self, image_path, save_result=False, result_path=None):
        """Analyze an image for golden ratio facial proportions"""
        try:
            # Detect landmarks
            landmarks, image = self.detect_landmarks(image_path)
            
            # Analyze facial proportions
            results = self.analyze_facial_proportions(landmarks)
            
            # Calculate overall score
            score = self.calculate_overall_score(results)
            
            # Generate visualization
            fig = self.visualize_analysis(image, landmarks, results, score)
            
            if save_result and result_path:
                plt.savefig(result_path)
                print(f"Result saved to {result_path}")
            
            plt.show()
            
            return {
                'measurements': results,
                'score': score
            }
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    analyzer = GoldenRatioFaceAnalyzer()
    result = analyzer.analyze_image('download.jpeg', save_result=True, result_path='golden_ratio_analysis.jpg')

    # result = analyzer.analyze_image('sample_face.jpg', save_result=True, result_path='golden_ratio_analysis.jpg')
      
    
    if result:
        print(f"Overall Golden Ratio Score: {result['score']:.2f}/100")
        print("\nDetailed Measurements:")
        for measurement, data in result['measurements'].items():
            print(f"{measurement.replace('_', ' ').title()}:")
            print(f"  Actual Ratio: {data['ratio']:.3f}")
            print(f"  Golden Ratio: {analyzer.GOLDEN_RATIO:.3f}")
            print(f"  Deviation: {data['deviation']:.2f}%")
            print()