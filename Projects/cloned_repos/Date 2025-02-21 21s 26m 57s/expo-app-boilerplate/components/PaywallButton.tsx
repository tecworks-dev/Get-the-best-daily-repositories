import { StyleSheet } from 'react-native';
import { TouchableOpacity } from 'react-native-gesture-handler';
import { ThemedText } from './ThemedText';
import { useSuperwall } from '@/hooks/useSuperwall';
import { SUPERWALL_TRIGGERS } from '@/config/superwall';

export function PaywallButton() {
  const { showPaywall } = useSuperwall();

  const handlePress = () => {
    showPaywall(SUPERWALL_TRIGGERS.ONBOARDING);
  };

  return (
    <TouchableOpacity style={styles.button} onPress={handlePress}>
      <ThemedText type="defaultSemiBold">Show Paywall</ThemedText>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    padding: 16,
    borderRadius: 8,
    backgroundColor: '#0a7ea4',
    alignItems: 'center',
  },
});