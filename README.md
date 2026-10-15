# sign-language-translator
Real-time ASL-to-text-and-speech translator using MediaPipe landmarks and sequence models (LSTM/Transformer).
A webcam-based system that recognizes ASL signs in real time, converts them to text, and speaks the result. Instead of classifying raw video, it extracts hand, pose and face landmarks with MediaPipe and feeds those keypoint sequences to a lightweight model. That makes it fast, runs on a laptop CPU, and is robust to lighting and background.
