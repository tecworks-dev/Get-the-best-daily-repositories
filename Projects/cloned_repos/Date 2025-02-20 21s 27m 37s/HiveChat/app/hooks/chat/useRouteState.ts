import { useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation'

const useRouteState = () => {
  const searchParams = useSearchParams();
  const [isFromHome, setIsFromHome] = useState(false);

  useEffect(() => {
    // 检查是否从首页跳转来的
    const checkRouteSource = () => {
      setIsFromHome(searchParams.get('f') === 'home');
    };

    checkRouteSource();
  }, [searchParams]);

  return isFromHome;
};

export default useRouteState;