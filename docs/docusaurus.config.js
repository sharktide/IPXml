// @ts-check

const { themes } = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;
import 'dotenv/config';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'IPXml',
  tagline: 'Declarative UI + preprocessing for ONNX apps',
  url: process.env.READTHEDOCS_CANONICAL_URL || 'https://localhost:3000',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  trailingSlash: true,
  organizationName: 'ipxml',
  projectName: 'ipxml',

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: "https://github.com/ipxml/ipxml/edit/main/docs/"
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css')
        }
      })
    ]
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'IPXml',
        items: [
          { type: 'docSidebar', sidebarId: 'main', position: 'left', label: 'Docs' },
          {
            href: 'https://github.com/your-org/ipxml',
            label: 'GitHub',
            position: 'right'
          }
        ]
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              { label: 'Getting Started', to: '/getting-started' },
              { label: 'Schema Reference', to: '/schema' }
            ]
          },
          {
            title: 'Examples',
            items: [
              { label: 'MNIST', to: '/examples/mnist' },
              { label: 'Inception v3', to: '/examples/inception-v3' }
            ]
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} IPXml`
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme
      }
    }),
    markdown: {
      hooks: {
        onBrokenMarkdownLinks: 'warn'
      }
    }
};

module.exports = config;
