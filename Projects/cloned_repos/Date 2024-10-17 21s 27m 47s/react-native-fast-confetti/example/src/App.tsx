import { useRef } from 'react';
import { StyleSheet, View, Button } from 'react-native';
import { Confetti } from 'react-native-fast-confetti';
import type { ConfettiMethods } from 'react-native-fast-confetti';

export default function App() {
  const confettiRef = useRef<ConfettiMethods>(null);

  return (
    <View style={styles.container}>
      <Confetti ref={confettiRef} />
      <Button title="Resume" onPress={() => confettiRef.current?.resume()} />
      <Button title="Pause" onPress={() => confettiRef.current?.pause()} />
      <Button title="Restart" onPress={() => confettiRef.current?.restart()} />
      <Button title="Reset" onPress={() => confettiRef.current?.reset()} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
