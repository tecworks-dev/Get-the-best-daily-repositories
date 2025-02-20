"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from 'next/link';
import { register } from '../actions';
import { fetchAppSettings } from '@/app/admin/system/actions';
import { Form, Input, Button, Alert } from 'antd';
import logo from "@/app/images/logo.png";
import Hivechat from "@/app/images/hivechat.svg";
import { useTranslations } from 'next-intl';

interface RegisterFormValues {
  email: string;
  password: string;
  repeatPassword: string;
}

export default function RegisterPage() {
  const t = useTranslations('Auth');
  const [form] = Form.useForm<RegisterFormValues>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isRegistrationOpen, setIsRegistrationOpen] = useState(false);
  const [fetchStatus, setFetchStatus] = useState(true);

  useEffect(() => {
    const fetchSettings = async () => {
      const resultValue = await fetchAppSettings('isRegistrationOpen');
      if (resultValue !== 'true') {
        setIsRegistrationOpen(false);
      } else {
        setIsRegistrationOpen(true);
        setFetchStatus(false);
      }
      setFetchStatus(false);
    }
    fetchSettings();
  }, []);

  async function handleSubmit(values: RegisterFormValues) {
    setLoading(true);
    if (values.password.length < 8 || values.repeatPassword.length < 8) {
      setError(t('passwordLengthLimit'));
      setLoading(false);
      return;
    }
    if (values.password !== values.repeatPassword) {
      setError(t('passwordNotSame'));
      setLoading(false);
      return;
    }
    try {
      const result = await register(values.email, values.password);
      if (result.status === 'success') {
        window.location.href = "/chat";
      } else {
        setError(result.message || t('registerFail'));
      }
      setLoading(false);
    } catch (err: any) {
      setError(err.message || t('registerFail'));
    }
  }
  if (fetchStatus) {
    return null;
  }

  if (!isRegistrationOpen) {
    return <main className="h-dvh flex justify-center items-center">
      <span className='ml-2 text-gray-600'>{t('openRegisterNotice')}</span>
    </main>
  }

  return (
    <div className="flex flex-col min-h-screen items-center justify-center bg-slate-50">
      <div className="flex items-center flex-row  mb-6">
        <Link href="/" className='flex items-center'>
          <Image src={logo} className="ml-1" alt="HiveChat logo" width={32} height={32} />
          <Hivechat className="ml-1" alt="HiveChat text" width={156} height={39} />
        </Link>
      </div>
      <div className="w-full max-w-sm space-y-6 rounded-lg bg-white p-6 shadow-xl">
        <h2 className="text-center text-2xl">{t('register')}</h2>
        {error && <Alert message={error} type="error" />}
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          requiredMark='optional'
        >
          <Form.Item
            name="email"
            label={<span className="font-medium">Email</span>}
            validateTrigger='onBlur'
            rules={[{ required: true, type: 'email', message: t('emailNotice') }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="password"
            label={<span className="font-medium">{t('password')}</span>}
            rules={[{ required: true, message: t('passwordNotice') }]}
          >
            <Input.Password />
          </Form.Item>
          <Form.Item
            name="repeatPassword"
            label={<span className="font-medium">{t('repeatPassword')}</span>}
            rules={[{ required: true, message: t('passwordNotice') }]}
          >
            <Input.Password />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
            >
              {t('registerShot')}
            </Button>
          </Form.Item>
          <div className='flex flex-row items-center -mt-2'>
            <span>{t('hasAccount')}</span>
            <Link href='/login'>
              <Button
                type='link'
                className='text-sm text-gray-400'
                style={{ 'padding': '0' }}
              >{t('clickToLogin')}</Button>
            </Link>
          </div>
        </Form>
      </div>
    </div>
  );
}