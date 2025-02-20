import React, { useState } from 'react';
import type { InputNumberProps } from 'antd';
import { Col, InputNumber, Row, Slider, Space } from 'antd';
const AdvancedSettingsPopover = () => {
  const [temperature, setTemperature] = useState(0);
  const [topp, setTopp] = useState(1);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0);
  const [presencePenalty, setPresencePenalty] = useState(0);

  const temperatureOnChange: InputNumberProps['onChange'] = (newValue) => {
    setTemperature(newValue as number);
  };
  const toppOnChange: InputNumberProps['onChange'] = (newValue) => {
    setTopp(newValue as number);
  };

  const frequencyPenaltyOnChange: InputNumberProps['onChange'] = (newValue) => {
    setFrequencyPenalty(newValue as number);
  };

  const presencePenaltyOnChange: InputNumberProps['onChange'] = (newValue) => {
    setPresencePenalty(newValue as number);
  };

  return (
    <div className='w-96 p-4'>
      <div className='flex justify-between'>
        <span className='font-medium'>Temperature</span>
        <InputNumber
          min={0}
          max={1}
          step={0.01}
          size='small'
          style={{ width: 50 }}
          value={temperature}
          onChange={temperatureOnChange}
        />
      </div>
      <Slider
        min={0}
        max={1}
        step={0.01}
        onChange={temperatureOnChange}
        value={typeof temperature === 'number' ? temperature : 1}
      />

      <div className='flex justify-between'>
        <span className='font-medium'>Top P</span>
        <InputNumber
          min={0.01}
          max={1}
          step={0.01}
          size='small'
          stringMode
          style={{ width: 50 }}
          value={topp}
          onChange={toppOnChange}
        />
      </div>
      <Slider
        min={0.01}
        max={1}
        step={0.01}
        onChange={toppOnChange}
        value={typeof topp === 'number' ? topp : 0}
      />
      
      <div className='flex justify-between'>
        <span className='font-medium'>Frequency Penalty</span>
        <InputNumber
          min={0}
          max={2}
          step={0.01}
          size='small'
          stringMode
          style={{ width: 50 }}
          value={frequencyPenalty}
          onChange={frequencyPenaltyOnChange}
        />
      </div>
      <Slider
        min={0}
        max={2}
        step={0.01}
        onChange={frequencyPenaltyOnChange}
        value={typeof frequencyPenalty === 'number' ? frequencyPenalty : 0}
      />
      
      <div className='flex justify-between'>
        <span className='font-medium'>Presence Penalty</span>
        <InputNumber
          min={0}
          max={2}
          step={0.01}
          size='small'
          stringMode
          style={{ width: 50 }}
          value={presencePenalty}
          onChange={presencePenaltyOnChange}
        />
      </div>
      <Slider
        min={0}
        max={2}
        step={0.01}
        onChange={presencePenaltyOnChange}
        value={typeof presencePenalty === 'number' ? presencePenalty : 0}
      />
    </div>
  )
}

export default AdvancedSettingsPopover