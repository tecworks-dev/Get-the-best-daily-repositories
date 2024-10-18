import React from 'react'
import { ConfigProvider } from 'antd'

const withTheme = (node: React.ReactNode) => (
  <>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#008080',
        },
        components: {
          Menu: {
            itemSelectedColor: '#008080',
            itemActiveBg: '#008080',
            itemHoverColor: '#008080'
          }
        }
      }}
    >
      {node}
    </ConfigProvider>
  </>
)

export default withTheme
