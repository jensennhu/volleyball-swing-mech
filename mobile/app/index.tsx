/** Record screen — camera view with 10-sec max recording. */

import { useState, useRef, useCallback, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from "react-native";
import { Video, ResizeMode } from "expo-av";
import { CameraView, useCameraPermissions, useMicrophonePermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import { useRouter } from "expo-router";
import { uploadVideo } from "../services/api";
import { useSpikeStore } from "../stores/spikeStore";

const MAX_DURATION_SEC = 10;

export default function RecordScreen() {
  const [camPerm, requestCamPerm] = useCameraPermissions();
  const [micPerm, requestMicPerm] = useMicrophonePermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(MAX_DURATION_SEC);
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const router = useRouter();
  const { addRecord } = useSpikeStore();

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = useCallback(async () => {
    if (!cameraRef.current) return;
    setIsRecording(true);
    setCountdown(MAX_DURATION_SEC);

    timerRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          cameraRef.current?.stopRecording();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: MAX_DURATION_SEC,
      });
      if (video?.uri) {
        setPreviewUri(video.uri);
      } else {
        Alert.alert("Recording Failed", "No video was captured. Please try again.");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown recording error";
      Alert.alert("Recording Error", msg);
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setIsRecording(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    cameraRef.current?.stopRecording();
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const pickVideo = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["videos"],
      allowsEditing: true,
      videoMaxDuration: MAX_DURATION_SEC,
      quality: 1,
    });
    if (!result.canceled && result.assets[0]?.uri) {
      setPreviewUri(result.assets[0].uri);
    }
  }, []);

  const retake = useCallback(() => {
    setPreviewUri(null);
  }, []);

  const analyzeVideo = useCallback(async () => {
    if (!previewUri || uploading) return;
    setUploading(true);

    try {
      const { video_id } = await uploadVideo(previewUri);

      addRecord({
        id: video_id,
        recordedAt: new Date().toISOString(),
        localUri: previewUri,
        metrics: {},
        status: "processing",
        serverUrl: "http://localhost:8000/api",
      });

      setPreviewUri(null);
      router.navigate("/history");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      Alert.alert("Error", msg);
    } finally {
      setUploading(false);
    }
  }, [previewUri, uploading, addRecord, router]);

  // Permission handling
  if (!camPerm || !micPerm) return <View style={styles.container} />;

  if (!camPerm.granted || !micPerm.granted) {
    const requestAll = async () => {
      if (!camPerm.granted) await requestCamPerm();
      if (!micPerm.granted) await requestMicPerm();
    };
    return (
      <View style={styles.container}>
        <Text style={styles.permText}>
          Camera and microphone access are needed to record spikes
        </Text>
        <TouchableOpacity style={styles.btn} onPress={requestAll}>
          <Text style={styles.btnText}>Grant Permissions</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Preview state — show video player + Analyze / Retake
  if (previewUri) {
    return (
      <View style={styles.previewScreen}>
        <Video
          source={{ uri: previewUri }}
          style={styles.previewVideo}
          resizeMode={ResizeMode.CONTAIN}
          useNativeControls
          shouldPlay={false}
        />
        {uploading ? (
          <View style={styles.uploadingRow}>
            <ActivityIndicator color="#6366f1" size="small" />
            <Text style={styles.uploadingText}>Uploading...</Text>
          </View>
        ) : (
          <View style={styles.previewActions}>
            <TouchableOpacity style={styles.btn} onPress={analyzeVideo}>
              <Text style={styles.btnText}>Analyze</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.btn, styles.btnOutline]} onPress={retake}>
              <Text style={styles.btnOutlineText}>Retake</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    );
  }

  // Camera view
  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        mode="video"
      >
        {isRecording && (
          <View style={styles.countdownBadge}>
            <Text style={styles.countdownText}>{countdown}s</Text>
          </View>
        )}
      </CameraView>

      <View style={styles.controls}>
        {isRecording ? (
          <TouchableOpacity style={styles.stopBtn} onPress={stopRecording}>
            <View style={styles.stopSquare} />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.recordBtn} onPress={startRecording}>
            <View style={styles.recordInner} />
          </TouchableOpacity>
        )}
        <Text style={styles.hint}>
          {isRecording ? "Tap to stop" : "Tap to record (10s max)"}
        </Text>
        {!isRecording && (
          <TouchableOpacity style={styles.uploadBtn} onPress={pickVideo}>
            <Text style={styles.uploadBtnText}>Upload Video</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0a0a0a",
    justifyContent: "center",
    alignItems: "center",
  },
  camera: {
    flex: 1,
    width: "100%",
  },
  controls: {
    paddingVertical: 24,
    alignItems: "center",
    backgroundColor: "#0a0a0a",
  },
  recordBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: "#fff",
    justifyContent: "center",
    alignItems: "center",
  },
  recordInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: "#ef4444",
  },
  stopBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: "#ef4444",
    justifyContent: "center",
    alignItems: "center",
  },
  stopSquare: {
    width: 28,
    height: 28,
    borderRadius: 4,
    backgroundColor: "#ef4444",
  },
  countdownBadge: {
    position: "absolute",
    top: 60,
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  countdownText: {
    color: "#ef4444",
    fontSize: 24,
    fontWeight: "700",
  },
  hint: {
    color: "#888",
    fontSize: 13,
    marginTop: 8,
  },
  uploadBtn: {
    marginTop: 16,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#444",
  },
  uploadBtnText: {
    color: "#aaa",
    fontSize: 14,
    fontWeight: "500",
  },
  permText: {
    color: "#ccc",
    fontSize: 16,
    marginBottom: 20,
    textAlign: "center",
    paddingHorizontal: 40,
  },
  btn: {
    backgroundColor: "#6366f1",
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 10,
  },
  btnText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  btnOutline: {
    backgroundColor: "transparent",
    borderWidth: 1,
    borderColor: "#555",
  },
  btnOutlineText: {
    color: "#aaa",
    fontSize: 16,
    fontWeight: "600",
  },
  // Preview screen
  previewScreen: {
    flex: 1,
    backgroundColor: "#000",
  },
  previewVideo: {
    flex: 1,
    width: "100%",
    backgroundColor: "#000",
  },
  uploadingRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    paddingVertical: 24,
    backgroundColor: "#0a0a0a",
  },
  uploadingText: {
    color: "#aaa",
    fontSize: 15,
  },
  previewActions: {
    flexDirection: "row",
    gap: 16,
    justifyContent: "center",
    paddingVertical: 24,
    backgroundColor: "#0a0a0a",
  },
});
