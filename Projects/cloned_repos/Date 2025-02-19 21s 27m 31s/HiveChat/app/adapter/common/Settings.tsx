'use client'
import Link from 'next/link';
import Markdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import React, { useState, useEffect } from 'react';
import { Button, Form, Input, Switch, Skeleton } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { getLLMInstance } from '@/app/adapter/models';
import Image from "next/image";
import { useTranslations } from 'next-intl';
import { saveToServer } from '@/app/adapter/actions';
import { fetchLlmModels } from '@/app/adapter/actions';
import useModelListStore from '@/app/store/modelList';
import CheckApiModal from '@/app/adapter/common/CheckApiModal';
import EditModelModal from '@/app/adapter/common/EditModelModal';
import AddModelModal from '@/app/adapter/common/AddModelModal';
import ModelList from '@/app/adapter/common/ModelList';
import { LLMModel } from '@/app/adapter/interface';
import { getLlmOriginConfigByProvider } from '@/app/utils/llms';



type FormValues = {
  status: boolean;
  apikey: string;
  endpoint: string;
}

const Settings = (props: { providerId: string }) => {
  const { allProviderList, modelList, initModelList, toggleProvider } = useModelListStore();
  const provider = allProviderList.find((i) => i.id === props.providerId)!;
  const t = useTranslations('Admin.Models');
  const [isClient, setIsClient] = useState(false);
  const [isPending, setIsPending] = useState(true);

  const [isCheckApiModalOpen, setIsCheckApiModalOpen] = useState(false);
  const [isCustomModelModalOpen, setIsCustomModelModalOpen] = useState(false);
  const [isEditModelModalOpen, setIsEditModelModalOpen] = useState(false);
  const [curretEditModal, setCurretEditModal] = useState<LLMModel>();
  const [checkResult, setCheckResult] = useState('init');
  const [errorMessage, setErrorMessage] = useState('');
  const [form] = Form.useForm();


  const getHelpLinks = (providerId: string) => {

    const helpLinks = {
      'openai': 'https://k2swpw8zgf.feishu.cn/wiki/J3FtwGumMi7k0vktB41cHvjTnTg',
      'claude': 'https://k2swpw8zgf.feishu.cn/wiki/XrMdwQlRKiOESdkdexMcQ89vn2e',
      'gemini': 'https://k2swpw8zgf.feishu.cn/wiki/WtEnw75PfiZ9LAkRXrqck6w5nsf',
      'moonshot': 'https://k2swpw8zgf.feishu.cn/wiki/ZqM3wCfDViRFxJkHGP4c9imsnig',
      'qwen': 'https://k2swpw8zgf.feishu.cn/wiki/EK10wXbHliNn3mkyKc9cCVJSnlb',
      'deepseek': 'https://k2swpw8zgf.feishu.cn/wiki/TgGqw09QNie4d1keCERcx4hBnhe',
      'volcengine': 'https://k2swpw8zgf.feishu.cn/wiki/PRuewmqAXi0Ykjk6gIBcPW4DnBe',
      'qianfan': 'https://k2swpw8zgf.feishu.cn/wiki/PUKvw62CgiZLoCkR9xjcWktinrc',
      'siliconflow': 'https://k2swpw8zgf.feishu.cn/wiki/EpD4wAj0ai681hkFFqMcvQZUn8g',
      'ollama': 'https://k2swpw8zgf.feishu.cn/wiki/MiPKw3uI7iS7ImkdAG7cRj77nuf',
    };
    const providers = Object.keys(helpLinks);
    if (providers.includes(providerId)) {
      return helpLinks[providerId as keyof typeof helpLinks];
    } else {
      return 'https://k2swpw8zgf.feishu.cn/wiki/J3FtwGumMi7k0vktB41cHvjTnTg';
    }
  }
  const chatbot = getLLMInstance(props.providerId)
  useEffect(() => {
    setIsClient(true);
  }, []);


  useEffect(() => {
    const fetchLlmConfig = async (): Promise<void> => {
      const result = await getLlmOriginConfigByProvider(provider.id);
      form.setFieldsValue({
        status: result.isActive || false,
        apikey: result.apikey || '',
        endpoint: result.endpoint || '',
      });
    };

    const fetchModelList = async (): Promise<void> => {
      const result = await fetchLlmModels(provider.id);
      initModelList(result)
    }

    const initData = async () => {
      try {
        await Promise.all([fetchLlmConfig(), fetchModelList()]);
        setIsPending(false);
      } catch (error) {
        setIsPending(false);
      }
    };

    initData();
  }, [form, initModelList, provider]);

  if (!isClient) return null;

  const onFinish = (values: FormValues) => {
    toggleProvider(provider.id, values.status);
    saveToServer(provider.id, { ...values, providerName: provider.providerName });
  };

  const checkApi = async (modelId: string) => {
    setCheckResult('pending')
    const result = await chatbot.check(modelId, form.getFieldValue('apikey'), form.getFieldValue('endpoint'));
    if (result.status === 'success') {
      setCheckResult('success');
      setErrorMessage('');
    } else {
      setCheckResult('error');
      setErrorMessage(result.message || '');
    }
  };

  return (
    isPending ? <Skeleton active style={{ 'marginTop': '1.5rem' }} /> :
      <div className='flex w-full flex-col'>
        <Form
          layout="vertical"
          form={form}
          onFinish={onFinish}
        >
          <div className='flex flex-row justify-between my-4 items-center'>
            <div className='flex items-center'>
              <Image src={provider.providerLogo!} alt="" width={26} height={26} />
              <h2 className='font-medium text-lg ml-2'>{provider.providerName}</h2>
            </div>
            <Form.Item name='status' style={{ 'margin': '0' }}>
              <Switch onChange={() => {
                form.submit();
              }} />
            </Form.Item>
          </div>
          <Form.Item label={<span className='font-medium'>API Key</span>} name='apikey'>
            <Input
              onBlur={() => {
                form.submit();
              }
              }
            />
          </Form.Item>
          <div className='felx text-sm text-gray-600 -mt-6 mb-2'>
            <Link
              href={getHelpLinks(props.providerId)}
              target='_blank'
            >
              <Button
                type='link'
                style={{ padding: '0' }}
              >
                {t('configGuide')}
              </Button>
            </Link>
          </div>
          {
            props.providerId === 'ollama' ?
              <Form.Item label={<span className='font-medium'>{t('serviceEndpoint')}</span>} name='endpoint'>
                <Input
                  type='url'
                  onBlur={() => {
                    form.submit();
                  }
                  }
                />
              </Form.Item> :
              <Form.Item label={<span className='font-medium'>{t('endpoint')} ({t('optional')})</span>} name='endpoint'>
                <Input
                  type='url'
                  onBlur={() => {
                    form.submit();
                  }
                  }
                />
              </Form.Item>
          }
        </Form>

        <div className='flex flex-col mb-2'>
          <div className='font-medium'>{t('testConnect')}</div>
          <div className='my-2 flex flex-row items-center'>
            {/* <Button loading={checkResult === 'pending'} onClick={checkApi}>{t('check')}</Button> */}
            <Button loading={checkResult === 'pending'} onClick={() => {
              setIsCheckApiModalOpen(true);
            }}>{t('check')}</Button>
            <div className='ml-2 flex flex-row items-center' >
              {checkResult === 'success' &&
                <>
                  <CheckCircleOutlined style={{ color: 'green', fontSize: '16px' }} />
                  <span className="text-green-700 ml-1 text-sm">{t('checkSuccess')}</span>
                </>}
              {checkResult === 'error' &&
                <>
                  <CloseCircleOutlined style={{ color: '#f87171', fontSize: '16px' }} />
                  <span className="text-red-400 ml-1 text-sm">{t('checkFail')}</span>
                </>}
            </div>
          </div>
        </div>
        {checkResult === 'error' &&
          <div className='overflow-x-auto mb-4 bg-red-50 rounded-md p-1 text-xs'>
            <div className='overflow-x-auto p-2 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-red-50 scrollbar-track-rounded-full scrollbar-thumb-rounded-full'>
              <Markdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[
                  [
                    rehypeHighlight,
                    {
                      detect: false,
                      ignoreMissing: true,
                    },
                  ],
                ]}
              >{errorMessage}</Markdown>
            </div>
          </div>
        }
        <ModelList
          setCurretEditModal={setCurretEditModal}
          setIsEditModelModalOpen={setIsEditModelModalOpen}
          setIsCustomModelModalOpen={setIsCustomModelModalOpen}
        />

        <CheckApiModal
          isCheckApiModalOpen={isCheckApiModalOpen}
          setIsCheckApiModalOpenOpen={setIsCheckApiModalOpen}
          startCheck={checkApi}
        />

        <AddModelModal
          isCustomModelModalOpen={isCustomModelModalOpen}
          setIsCustomModelModalOpen={setIsCustomModelModalOpen}
          providerId={provider.id}
          providerName={provider.providerName}
        />
        <EditModelModal
          model={curretEditModal}
          isEditModelModalOpen={isEditModelModalOpen}
          setIsEditModelModalOpen={setIsEditModelModalOpen}
          providerId={provider.id}
          providerName={provider.providerName}
        />
      </div>
  );
};

export default Settings;
