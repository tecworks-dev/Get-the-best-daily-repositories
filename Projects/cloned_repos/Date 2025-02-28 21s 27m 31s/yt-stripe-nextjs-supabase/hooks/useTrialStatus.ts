import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/utils/supabase';

export function useTrialStatus() {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [trialStatus, setTrialStatus] = useState<{
    isInTrial: boolean;
    trialEndTime: string | null;
  }>({ isInTrial: false, trialEndTime: null });

  useEffect(() => {
    async function checkTrialStatus() {
      if (!user?.id) {
        setIsLoading(false);
        return;
      }

      try {
        // First check if user has an active subscription
        const { data: subscription } = await supabase
          .from('subscriptions')
          .select('status')
          .eq('user_id', user.id)
          .maybeSingle();

        // If user has an active subscription, skip trial creation
        if (subscription?.status === 'active' || subscription?.status === 'trialing') {
          setTrialStatus({
            isInTrial: false,
            trialEndTime: null
          });
          setIsLoading(false);
          return;
        }

        // Check if user has an existing trial
        const { data: trial, error: trialError } = await supabase
          .from('user_trials')
          .select('trial_end_time, is_trial_used')
          .eq('user_id', user.id)
          .maybeSingle();

        if (trialError && trialError.code !== 'PGRST116') { // PGRST116 is "not found" error
          throw trialError;
        }

        if (trial) {
          // Check if trial is still valid
          const now = new Date();
          const endTime = new Date(trial.trial_end_time);
          const isInTrial = !trial.is_trial_used && now < endTime;

          setTrialStatus({
            isInTrial,
            trialEndTime: trial.trial_end_time
          });
        } else {
          // Create new trial for user
          const trialEndTime = new Date();
          trialEndTime.setHours(trialEndTime.getHours() + 48);

          const { data: newTrial, error: insertError } = await supabase
            .from('user_trials')
            .upsert({ // Use upsert instead of insert to handle duplicates
              user_id: user.id,
              trial_end_time: trialEndTime.toISOString(),
              is_trial_used: false
            })
            .select('trial_end_time')
            .single();

          if (insertError) throw insertError;

          setTrialStatus({
            isInTrial: true,
            trialEndTime: newTrial.trial_end_time
          });
        }
      } catch (error) {
        console.error('Error checking trial status:', error);
        // Set default state on error
        setTrialStatus({
          isInTrial: false,
          trialEndTime: null
        });
      } finally {
        setIsLoading(false);
      }
    }

    checkTrialStatus();
  }, [user?.id]);

  return { ...trialStatus, isLoading };
}