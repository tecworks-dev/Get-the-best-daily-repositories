"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import Link from 'next/link';
import { Form, Input, Button, Alert } from 'antd';
import logo from "@/app/images/logo.png";
import Hivechat from "@/app/images/hivechat.svg";
import { useTranslations } from 'next-intl';

interface LoginFormValues {
  email: string;
  password: string;
}

export default function LoginPage() {
  const t = useTranslations('Auth');
  const [form] = Form.useForm<LoginFormValues>();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(values: LoginFormValues) {
    setLoading(true);
    const response = await signIn("credentials", {
      email: values.email,
      password: values.password,
      redirect: false,
    });
    setLoading(false);
    if (response?.error) {
      console.log(response?.error);
      setError(t('passwordError'));
      return;
    }
    router.push("/chat");
  }

  return (
    <div className="flex flex-col min-h-screen items-center justify-center bg-slate-50">
      <div className="flex items-center flex-row  mb-6">
        <Link href="/" className='flex items-center'>
          <Image src={logo} className="ml-1" alt="HiveChat logo" width={32} height={32} />
          <Hivechat className="ml-1" alt="HiveChat text" width={156} height={39} />
        </Link>
      </div>
      <div className="w-full max-w-md space-y-6 rounded-lg bg-white p-8 shadow-xl">
        <h2 className="text-center text-2xl">{t('login')}</h2>
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
            <Input size='large' />
          </Form.Item>
          <Form.Item
            name="password"
            label={<span className="font-medium">{t('password')}</span>}
            rules={[{ required: true, message: t('passwordNotice') }]}
          >
            <Input.Password size='large' />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
              size='large'
            >
              {t('login')}
            </Button>
          </Form.Item>
          <div className='flex -mt-2'>
            <Link href='/register'>
              <Button
                type='link'
                className='text-sm text-gray-400'
                style={{ 'padding': '0' }}
              >{t('register')}</Button>
            </Link>
          </div>
        </Form>
      </div>
    </div>
  );
}